use anyhow::{bail, Context, Result};

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
    pub limits: wgpu::Limits,
    pub features: wgpu::Features,
    /// Bytes staged by `write_buffer` since the last drain. It is cumulative
    /// across calls on purpose: `write_buffer` keeps the host copy until the
    /// queue drains, and a per-call counter lets the small per-tensor writes
    /// accumulate the whole model (~900 MiB) before anything is polled.
    pending: std::cell::Cell<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum DeviceSelector {
    #[default]
    Auto,
    Runtime { api: wgpu::Backend, index: usize },
    Cpu,
    Index(usize),
    Name(String),
}

const RUNTIMES: &[(&str, wgpu::Backend)] = &[
    ("vulkan", wgpu::Backend::Vulkan),
    ("metal", wgpu::Backend::Metal),
    ("dx12", wgpu::Backend::Dx12),
    ("d3d12", wgpu::Backend::Dx12),
    ("gl", wgpu::Backend::Gl),
];

const API_RANK_ORDER: [wgpu::Backend; 4] = [
    wgpu::Backend::Vulkan,
    wgpu::Backend::Metal,
    wgpu::Backend::Dx12,
    wgpu::Backend::Gl,
];

impl DeviceSelector {
    pub fn parse(spec: &str) -> Result<Self> {
        let s = spec.trim();
        if s.is_empty() || s.eq_ignore_ascii_case("auto") {
            return Ok(Self::Auto);
        }
        if s.eq_ignore_ascii_case("cpu") {
            return Ok(Self::Cpu);
        }
        if s.eq_ignore_ascii_case("gpu") {
            bail!("`gpu` 不是适配器名字。用 auto、vulkan、dx12，或适配器名称的一部分");
        }
        if let Some(rest) = s.strip_prefix('#') {
            return Ok(Self::Index(rest.trim().parse().context("device index")?));
        }
        if let Ok(i) = s.parse::<usize>() {
            return Ok(Self::Index(i));
        }
        let (name, index) = match s.split_once(':') {
            Some((n, i)) => (n, i.trim().parse().context("runtime device index")?),
            None => (s, 0usize),
        };
        if let Some((_, api)) = RUNTIMES.iter().find(|(n, _)| name.eq_ignore_ascii_case(n)) {
            return Ok(Self::Runtime { api: *api, index });
        }
        Ok(Self::Name(s.to_lowercase()))
    }

    fn matches(&self, info: &wgpu::AdapterInfo) -> bool {
        match self {
            Self::Auto => true,
            Self::Name(n) => info.name.to_lowercase().contains(n),
            Self::Index(_) | Self::Cpu => true,
            Self::Runtime { api, .. } => info.backend == *api,
        }
    }
}

fn device_class(kind: wgpu::DeviceType) -> u8 {
    match kind {
        wgpu::DeviceType::DiscreteGpu => 0,
        wgpu::DeviceType::IntegratedGpu => 1,
        wgpu::DeviceType::VirtualGpu => 2,
        wgpu::DeviceType::Cpu => 3,
        _ => 4,
    }
}

fn rank(info: &wgpu::AdapterInfo) -> (u8, u8) {
    let class = device_class(info.device_type);
    let api = match info.backend {
        wgpu::Backend::Vulkan => 0,
        wgpu::Backend::Metal => 1,
        wgpu::Backend::Dx12 => 2,
        wgpu::Backend::Gl => 3,
        _ => 4,
    };
    (class, api)
}

fn instance_for(backends: wgpu::Backends) -> wgpu::Instance {
    if backends == wgpu::Backends::all() {
        return wgpu::Instance::default();
    }
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = backends;
    wgpu::Instance::new(desc)
}

async fn adapters_for(selector: &DeviceSelector) -> Vec<wgpu::Adapter> {
    match selector {
        DeviceSelector::Runtime { api, .. } => {
            let b = wgpu::Backends::from(*api);
            instance_for(b).enumerate_adapters(b).await
        }
        // A name match tries Vulkan first: creating an instance over *every*
        // backend costs ~585 MiB of driver working set on this machine
        // (DX12/GL bring up their own runtimes), against ~55 MiB for Vulkan
        // alone.  If nothing matches on Vulkan the caller falls back below.
        DeviceSelector::Name(_) => {
            let vulkan = instance_for(wgpu::Backends::VULKAN)
                .enumerate_adapters(wgpu::Backends::VULKAN)
                .await;
            if vulkan
                .iter()
                .any(|a| selector.matches(&a.get_info()) && device_class(a.get_info().device_type) <= 2)
            {
                vulkan
            } else {
                instance_for(wgpu::Backends::all())
                    .enumerate_adapters(wgpu::Backends::all())
                    .await
            }
        }
        DeviceSelector::Index(_) => {
            instance_for(wgpu::Backends::all())
                .enumerate_adapters(wgpu::Backends::all())
                .await
        }
        DeviceSelector::Auto => {
            let mut all = Vec::new();
            for api in API_RANK_ORDER {
                let b = wgpu::Backends::from(api);
                let mut found = instance_for(b).enumerate_adapters(b).await;
                let discrete = found
                    .iter()
                    .any(|a| a.get_info().device_type == wgpu::DeviceType::DiscreteGpu);
                all.append(&mut found);
                if discrete {
                    break;
                }
            }
            all
        }
        DeviceSelector::Cpu => Vec::new(),
    }
}

fn list_names(adapters: &[wgpu::Adapter]) -> String {
    adapters
        .iter()
        .enumerate()
        .map(|(i, a)| {
            let info = a.get_info();
            format!("#{i} {} ({:?}, {:?})", info.name, info.backend, info.device_type)
        })
        .collect::<Vec<_>>()
        .join(", ")
}

pub fn list_devices() -> Result<Vec<String>> {
    pollster::block_on(async {
        let adapters = instance_for(wgpu::Backends::all())
            .enumerate_adapters(wgpu::Backends::all())
            .await;
        if adapters.is_empty() {
            bail!("没有可见的 wgpu 适配器");
        }
        Ok(adapters
            .iter()
            .enumerate()
            .map(|(i, a)| {
                let info = a.get_info();
                let lim = a.limits();
                format!(
                    "#{i} {} ({:?}, {:?}) binding {} MiB",
                    info.name,
                    info.backend,
                    info.device_type,
                    lim.max_storage_buffer_binding_size / (1024 * 1024)
                )
            })
            .collect())
    })
}

impl Gpu {
    pub async fn new_with(selector: DeviceSelector) -> Result<Self> {
        if selector == DeviceSelector::Cpu {
            bail!("cpu 是主机后端，不是 wgpu 设备");
        }
        let adapters = adapters_for(&selector).await;
        if adapters.is_empty() {
            bail!("没有 wgpu 适配器（{selector:?}）");
        }
        let candidates: Vec<&wgpu::Adapter> = match &selector {
            DeviceSelector::Index(i) => vec![adapters.get(*i).ok_or_else(|| {
                anyhow::anyhow!(
                    "设备 #{i} 不存在（可见 {} 个：{}）",
                    adapters.len(),
                    list_names(&adapters)
                )
            })?],
            sel => {
                let mut hits: Vec<&wgpu::Adapter> = adapters
                    .iter()
                    .filter(|a| sel.matches(&a.get_info()))
                    .filter(|a| {
                        !matches!(sel, DeviceSelector::Auto)
                            || device_class(a.get_info().device_type) <= 2
                    })
                    .collect();
                if hits.is_empty() {
                    bail!("没有匹配 {sel:?} 的适配器（可见：{}）", list_names(&adapters));
                }
                hits.sort_by_key(|a| rank(&a.get_info()));
                match sel {
                    DeviceSelector::Runtime { index, .. } => {
                        vec![hits.get(*index).copied().ok_or_else(|| {
                            anyhow::anyhow!("该后端有 {} 个设备，下标 {index} 越界", hits.len())
                        })?]
                    }
                    DeviceSelector::Auto => hits,
                    _ => vec![hits[0]],
                }
            }
        };

        let mut opened = None;
        let mut skipped: Option<anyhow::Error> = None;
        // Per-dispatch GPU timing is diagnostic only, so the optional feature is
        // requested just for `SVS_GPU_PROF=1` runs and stays empty otherwise.
        let want_prof = crate::tuning::gpu_prof();
        for adapter in &candidates {
            let info = adapter.get_info();
            let limits = adapter.limits();
            let required_features = if want_prof
                && adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY)
            {
                wgpu::Features::TIMESTAMP_QUERY
            } else {
                wgpu::Features::empty()
            };
            match adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("sensevoice-small-wgpu"),
                    required_features,
                    required_limits: limits.clone(),
                    ..Default::default()
                })
                .await
            {
                Ok((device, queue)) => {
                    opened = Some((info, limits, device, queue));
                    break;
                }
                Err(e) => {
                    skipped = Some(e.into());
                }
            }
        }
        let Some((info, limits, device, queue)) = opened else {
            return Err(skipped.context("request_device")?);
        };
        if limits.max_compute_workgroup_storage_size < crate::gemm_f32::GEMM_SMEM {
            bail!(
                "{} 的 workgroup 共享内存上限 {} 字节，GEMM 需要 {}",
                info.name,
                limits.max_compute_workgroup_storage_size,
                crate::gemm_f32::GEMM_SMEM
            );
        }
        let feats = device.features();
        Ok(Self {
            device,
            queue,
            info,
            limits,
            features: feats,
            pending: std::cell::Cell::new(0),
        })
    }

    pub fn describe(&self) -> String {
        format!(
            "{} ({:?}, {:?}) | {} {} | binding {} MiB",
            self.info.name,
            self.info.backend,
            self.info.device_type,
            self.info.driver,
            self.info.driver_info,
            self.limits.max_storage_buffer_binding_size / (1024 * 1024)
        )
    }

    pub fn storage(&self, label: &str, bytes: u64) -> wgpu::Buffer {
        let size = (bytes + 255) & !255;
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.max(256),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    pub fn uniform(&self, label: &str, bytes: u64) -> wgpu::Buffer {
        let size = (bytes + 255) & !255;
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.max(256),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    pub fn write_at(&self, buf: &wgpu::Buffer, offset: u64, data: &[u8]) {
        self.queue.write_buffer(buf, offset, data);
    }

    pub fn upload_chunked(&self, buf: &wgpu::Buffer, data: &[u8]) -> Result<()> {
        const CHUNK: usize = 8 << 20;
        for (i, piece) in data.chunks(CHUNK).enumerate() {
            self.queue.write_buffer(buf, (i * CHUNK) as u64, piece);
            let pending = self.pending.get() + piece.len();
            if pending >= CHUNK {
                // Poll, not just submit: `write_buffer` stages host memory and
                // the copy only retires when the queue drains. Uploading is a
                // load-time cost, so the wait is free and bounds the staging.
                self.flush()?;
                self.pending.set(0);
            } else {
                self.pending.set(pending);
            }
        }
        Ok(())
    }

    pub fn pump(&self) -> Result<()> {
        let enc = self.device.create_command_encoder(&Default::default());
        self.queue.submit([enc.finish()]);
        Ok(())
    }

    pub fn flush(&self) -> Result<()> {
        self.pump()?;
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("poll")?;
        Ok(())
    }

    pub fn readback(&self, buf: &wgpu::Buffer, bytes: u64) -> Result<Vec<u8>> {
        let size = (bytes + 3) & !3;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: size.max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(buf, 0, &staging, 0, size.max(4));
        self.queue.submit([enc.finish()]);
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("poll readback")?;
        rx.recv().context("map callback")?.context("map buffer")?;
        let data = slice.get_mapped_range()?;
        let out = data[..bytes as usize].to_vec();
        drop(data);
        staging.unmap();
        Ok(out)
    }

    /// Copy several buffer prefixes into one staging buffer and map it once.
    ///
    /// Chaining several [`Self::readback`] calls submits and maps repeatedly;
    /// some drivers refuse the later maps ("failed to async map a buffer").
    /// One encoder plus one map avoids that and is what the debug dump uses.
    pub fn readback_many(&self, parts: &[(&wgpu::Buffer, u64)]) -> Result<Vec<u8>> {
        let sizes: Vec<u64> = parts
            .iter()
            .map(|(_, n)| (((*n).max(4)) + 3) & !3)
            .collect();
        let total: u64 = sizes.iter().sum();
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback-many"),
            size: total.max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        let mut off = 0u64;
        for ((buf, _), size) in parts.iter().zip(sizes.iter()) {
            enc.copy_buffer_to_buffer(buf, 0, &staging, off, *size);
            off += *size;
        }
        self.queue.submit([enc.finish()]);
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("poll readback-many")?;
        rx.recv().context("map callback")?.context("map buffer")?;
        let data = slice.get_mapped_range()?;
        let out = data.to_vec();
        drop(data);
        staging.unmap();
        Ok(out)
    }

    pub fn pipeline(
        &self,
        label: &str,
        wgsl: &str,
        entry: &str,
        layout: &wgpu::PipelineLayout,
    ) -> Result<wgpu::ComputePipeline> {
        let guard = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        let pipe = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        });
        if let Some(e) = pollster::block_on(guard.pop()) {
            bail!("pipeline {label} 校验失败: {e}");
        }
        Ok(pipe)
    }
}
