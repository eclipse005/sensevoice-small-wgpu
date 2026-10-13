//! Process affinity used to make the `gemm` crate's tiling deterministic.
//!
//! `gemm` reads CPUID leaf 4 lazily, so `kc` — and therefore the f32
//! accumulation order — depends on the core class the first call happens to
//! run on (P-core vs E-core L1d geometry on a hybrid CPU). The fix is to pin,
//! let the lazy init happen on a chosen core class, then unpin. See
//! `encoder_cpu::freeze_gemm_tiling` for the full story.

#[cfg(all(windows, target_arch = "x86_64"))]
mod win_affinity {
    #[link(name = "kernel32")]
    extern "system" {
        pub fn GetCurrentThread() -> *mut core::ffi::c_void;
        pub fn SetThreadAffinityMask(thread: *mut core::ffi::c_void, mask: usize) -> usize;
    }
}

/// Pin to the logical processor with the extreme L1d size (`prefer_small` picks
/// the minimum, otherwise the maximum) and return the previous affinity mask so
/// the caller can restore it. 0 = nothing to restore.
#[cfg(all(windows, target_arch = "x86_64"))]
pub(crate) fn pin_to_l1_extreme(prefer_small: bool) -> usize {
    use win_affinity as w;
    // SAFETY: GetCurrentThread returns a pseudo-handle; SetThreadAffinityMask
    // either rejects the mask (0) or applies it and returns the previous one.
    unsafe {
        let me = w::GetCurrentThread();
        let mut saved = 0usize;
        let mut best: Option<(usize, u64)> = None;
        for lp in 0..64usize {
            let prev = w::SetThreadAffinityMask(me, 1usize << lp);
            if prev == 0 {
                continue;
            }
            if saved == 0 {
                saved = prev;
            }
            let Some(size) = l1d_bytes() else { continue };
            if best.map_or(true, |(_, b)| {
                if prefer_small {
                    size < b
                } else {
                    size > b
                }
            }) {
                best = Some((lp, size));
            }
        }
        if let Some((lp, _)) = best {
            w::SetThreadAffinityMask(me, 1usize << lp);
        }
        saved
    }
}

#[cfg(not(all(windows, target_arch = "x86_64")))]
pub(crate) fn pin_to_l1_extreme(_prefer_small: bool) -> usize {
    0
}

pub(crate) fn unpin(saved: usize) {
    #[cfg(all(windows, target_arch = "x86_64"))]
    if saved != 0 {
        // SAFETY: `saved` is a mask this thread held before (see above).
        unsafe {
            win_affinity::SetThreadAffinityMask(win_affinity::GetCurrentThread(), saved);
        }
    }
    #[cfg(not(all(windows, target_arch = "x86_64")))]
    let _ = saved;
}

/// L1 data cache size of the core this thread is currently running on, from
/// CPUID leaf 4 — the same leaf `gemm` reads. `None` if leaf 4 says nothing.
#[cfg(all(windows, target_arch = "x86_64"))]
fn l1d_bytes() -> Option<u64> {
    for sub in 0..16u32 {
        // CPUID is always available on x86_64, and `__cpuid_count` is a safe
        // intrinsic there since Rust 1.87.
        let r = std::arch::x86_64::__cpuid_count(4, sub);
        let cache_type = r.eax & 0x1f;
        if cache_type == 0 {
            break;
        }
        let level = (r.eax >> 5) & 0x7;
        // 1 = data, 2 = unified. Skip the instruction cache.
        if level == 1 && (cache_type == 1 || cache_type == 2) {
            let line = u64::from(r.ebx & 0xfff) + 1;
            let parts = u64::from((r.ebx >> 12) & 0x3ff) + 1;
            let ways = u64::from((r.ebx >> 22) & 0x3ff) + 1;
            let sets = u64::from(r.ecx) + 1;
            return Some(line * parts * ways * sets);
        }
    }
    None
}
