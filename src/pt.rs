//! Official `model.pt`: a zip of pickle protocol 2 plus raw f32 storages.
//! Every tensor in SenseVoiceSmall is a contiguous `FloatStorage` at offset 0.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::rc::Rc;

use anyhow::{bail, Context, Result};

pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

pub fn load_state_dict(path: &Path) -> Result<HashMap<String, Tensor>> {
    let file = File::open(path).with_context(|| format!("打开 {}", path.display()))?;
    let mut zip = zip::ZipArchive::new(file).with_context(|| format!("不是 zip：{}", path.display()))?;
    let pkl = read_entry(&mut zip, "model/data.pkl")?;
    let records = unpickle(&pkl).context("解析 model.pt")?;
    let mut out = HashMap::with_capacity(records.len());
    for (name, rec) in records {
        let bytes = read_entry(&mut zip, &format!("model/data/{}", rec.key))?;
        let numel = rec.shape.iter().product::<usize>();
        if bytes.len() != rec.storage_numel * 4 || rec.storage_numel < numel {
            bail!(
                "{name} 存储 {} 字节，shape {:?} 需要至少 {} 个 f32",
                bytes.len(),
                rec.shape,
                numel
            );
        }
        let mut data = Vec::with_capacity(numel);
        for c in bytes[..numel * 4].chunks_exact(4) {
            data.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
        }
        out.insert(name, Tensor { shape: rec.shape, data });
    }
    Ok(out)
}

struct Rec {
    key: String,
    shape: Vec<usize>,
    storage_numel: usize,
}

fn read_entry(zip: &mut zip::ZipArchive<File>, name: &str) -> Result<Vec<u8>> {
    let mut file = zip
        .by_name(name)
        .with_context(|| format!("model.pt 缺少 {name}"))?;
    let mut buf = Vec::with_capacity(file.size() as usize);
    file.read_to_end(&mut buf)?;
    Ok(buf)
}

#[derive(Clone)]
enum V {
    Int(i64),
    Str(Rc<str>),
    False,
    Mark,
    Seq(Rc<[V]>),
    Dict(Rc<RefCell<HashMap<String, V>>>),
    Global(&'static str),
    Storage { key: Rc<str>, numel: i64 },
    Tensor { key: Rc<str>, shape: Rc<[usize]>, storage_numel: usize },
}

fn unpickle(data: &[u8]) -> Result<HashMap<String, Rec>> {
    let mut i = 0usize;
    let mut stack: Vec<V> = Vec::new();
    let mut memo: HashMap<u32, V> = HashMap::new();
    let need = |data: &[u8], i: usize, n: usize| -> Result<()> {
        if i + n > data.len() {
            bail!("model.pt pickle 在 {i} 处截断");
        }
        Ok(())
    };
    while i < data.len() {
        let op = data[i];
        i += 1;
        match op {
            0x80 => {
                need(data, i, 1)?;
                let proto = data[i];
                i += 1;
                if proto != 2 {
                    bail!("model.pt pickle 协议 {proto}，只支持 2");
                }
            }
            b'c' => {
                let module = read_cstr(data, &mut i)?;
                let name = read_cstr(data, &mut i)?;
                let g = match (module.as_str(), name.as_str()) {
                    ("collections", "OrderedDict") => "OrderedDict",
                    ("torch", "FloatStorage") => "FloatStorage",
                    ("torch._utils", "_rebuild_tensor_v2") => "rebuild",
                    _ => bail!("model.pt 出现未支持的类型 {module}.{name}"),
                };
                stack.push(V::Global(g));
            }
            b')' => stack.push(V::Seq(Rc::from([]))),
            b'}' => stack.push(V::Dict(Rc::new(RefCell::new(HashMap::new())))),
            b'(' => stack.push(V::Mark),
            0x89 => stack.push(V::False),
            b'K' => {
                need(data, i, 1)?;
                stack.push(V::Int(data[i] as i64));
                i += 1;
            }
            b'M' => {
                need(data, i, 2)?;
                stack.push(V::Int(u16::from_le_bytes([data[i], data[i + 1]]) as i64));
                i += 2;
            }
            b'J' => {
                need(data, i, 4)?;
                let n = i32::from_le_bytes(data[i..i + 4].try_into().unwrap());
                stack.push(V::Int(n as i64));
                i += 4;
            }
            b'X' => {
                need(data, i, 4)?;
                let n = u32::from_le_bytes(data[i..i + 4].try_into().unwrap()) as usize;
                i += 4;
                need(data, i, n)?;
                let s = String::from_utf8(data[i..i + n].to_vec()).context("pickle 字符串不是 utf-8")?;
                i += n;
                stack.push(V::Str(Rc::from(s)));
            }
            b'q' => {
                need(data, i, 1)?;
                let id = data[i] as u32;
                i += 1;
                memo.insert(id, stack.last().context("BINPUT 空栈")?.clone());
            }
            b'r' => {
                need(data, i, 4)?;
                let id = u32::from_le_bytes(data[i..i + 4].try_into().unwrap());
                i += 4;
                memo.insert(id, stack.last().context("LONG_BINPUT 空栈")?.clone());
            }
            b'h' => {
                need(data, i, 1)?;
                let id = data[i] as u32;
                i += 1;
                stack.push(memo.get(&id).with_context(|| format!("memo {id} 不存在"))?.clone());
            }
            b'j' => {
                need(data, i, 4)?;
                let id = u32::from_le_bytes(data[i..i + 4].try_into().unwrap());
                i += 4;
                stack.push(memo.get(&id).with_context(|| format!("memo {id} 不存在"))?.clone());
            }
            b't' | 0x85 | 0x86 | 0x87 => {
                let n = match op {
                    0x85 => 1,
                    0x86 => 2,
                    0x87 => 3,
                    _ => {
                        let mark = stack.iter().rposition(|v| matches!(v, V::Mark)).context("TUPLE 没有 MARK")?;
                        let items: Vec<V> = stack.drain(mark + 1..).collect();
                        stack.pop();
                        stack.push(V::Seq(Rc::from(items)));
                        continue;
                    }
                };
                if stack.len() < n {
                    bail!("TUPLE 元素不足");
                }
                let items: Vec<V> = stack.split_off(stack.len() - n);
                stack.push(V::Seq(Rc::from(items)));
            }
            b'R' => {
                let args = pop(&mut stack)?;
                let callable = pop(&mut stack)?;
                stack.push(reduce(callable, args)?);
            }
            b'Q' => {
                let id = pop(&mut stack)?;
                stack.push(persistent(id)?);
            }
            b's' => {
                let val = pop(&mut stack)?;
                let key = pop(&mut stack)?;
                let dict = stack.last().context("SETITEM 没有字典")?;
                dict_insert(dict, key, val)?;
            }
            b'u' => {
                let mark = stack.iter().rposition(|v| matches!(v, V::Mark)).context("SETITEMS 没有 MARK")?;
                let items: Vec<V> = stack.drain(mark + 1..).collect();
                stack.pop();
                if items.len() % 2 != 0 {
                    bail!("SETITEMS 键值不成对");
                }
                let dict = stack.last().context("SETITEMS 没有字典")?.clone();
                for pair in items.chunks_exact(2) {
                    dict_insert(&dict, pair[0].clone(), pair[1].clone())?;
                }
            }
            b'b' => {
                pop(&mut stack)?;
            }
            b'.' => break,
            other => bail!("model.pt pickle 不认识的操作 0x{other:02x}，位置 {}", i - 1),
        }
    }
    let top = stack.pop().context("model.pt 是空的")?;
    let V::Dict(map) = top else {
        bail!("model.pt 顶层不是状态字典");
    };
    let map = map.borrow();
    let mut out = HashMap::with_capacity(map.len());
    for (name, val) in map.iter() {
        let V::Tensor { key, shape, storage_numel } = val else {
            bail!("{name} 不是张量");
        };
        out.insert(
            name.clone(),
            Rec {
                key: key.to_string(),
                shape: shape.to_vec(),
                storage_numel: *storage_numel,
            },
        );
    }
    Ok(out)
}

fn reduce(callable: V, args: V) -> Result<V> {
    let V::Global(name) = callable else {
        bail!("REDUCE 的目标不是全局对象");
    };
    let V::Seq(items) = args else {
        bail!("REDUCE 参数不是元组");
    };
    match name {
        "OrderedDict" => {
            if !items.is_empty() {
                bail!("OrderedDict 参数不为空");
            }
            Ok(V::Dict(Rc::new(RefCell::new(HashMap::new()))))
        }
        "rebuild" => {
            if items.len() < 3 {
                bail!("_rebuild_tensor_v2 参数不足");
            }
            let V::Storage { key, numel } = &items[0] else {
                bail!("张量存储不是 FloatStorage");
            };
            let V::Int(offset) = items[1] else {
                bail!("张量 offset 不是整数");
            };
            if offset != 0 {
                bail!("张量 offset {offset}，只支持 0");
            }
            let V::Seq(shape_v) = &items[2] else {
                bail!("张量 shape 不是元组");
            };
            let mut shape = Vec::with_capacity(shape_v.len());
            for d in shape_v.iter() {
                let V::Int(n) = d else {
                    bail!("shape 维度不是整数");
                };
                if *n < 0 {
                    bail!("shape 维度 {n}");
                }
                shape.push(*n as usize);
            }
            let got = shape.iter().product::<usize>() as i64;
            // PyTorch may pad the storage past the tensor. The view starts at offset 0.
            if *numel < got {
                bail!("存储长度 {numel} 小于 shape {shape:?}");
            }
            if items.len() > 3 {
                let V::Seq(stride) = &items[3] else {
                    bail!("stride 不是元组");
                };
                let mut expect = Vec::with_capacity(shape.len());
                let mut acc = 1usize;
                for &dim in shape.iter().rev() {
                    expect.push(acc);
                    acc *= dim;
                }
                expect.reverse();
                let got: Vec<usize> = stride
                    .iter()
                    .map(|v| match v {
                        V::Int(n) if *n >= 0 => Ok(*n as usize),
                        _ => bail!("stride 不是整数"),
                    })
                    .collect::<Result<_>>()?;
                if got != expect {
                    bail!("张量不是行主序 contiguous，stride {got:?} shape {shape:?}");
                }
            }
            Ok(V::Tensor {
                key: key.clone(),
                shape: Rc::from(shape),
                storage_numel: *numel as usize,
            })
        }
        other => bail!("不支持 REDUCE {other}"),
    }
}

fn persistent(id: V) -> Result<V> {
    let V::Seq(items) = id else {
        bail!("BINPERSID 不是元组");
    };
    if items.len() != 5 {
        bail!("storage 记录长度 {}", items.len());
    }
    let V::Str(kind) = &items[0] else {
        bail!("storage 标记不是字符串");
    };
    if kind.as_ref() != "storage" {
        bail!("persistent 类型 {kind}");
    }
    let V::Global("FloatStorage") = &items[1] else {
        bail!("权重不是 FloatStorage");
    };
    let V::Str(key) = &items[2] else {
        bail!("storage 键不是字符串");
    };
    let V::Int(numel) = items[4] else {
        bail!("storage 长度不是整数");
    };
    Ok(V::Storage {
        key: key.clone(),
        numel,
    })
}

fn dict_insert(dict: &V, key: V, val: V) -> Result<()> {
    let V::Dict(map) = dict else {
        bail!("SETITEM 目标不是字典");
    };
    let V::Str(k) = key else {
        bail!("字典键不是字符串");
    };
    map.borrow_mut().insert(k.to_string(), val);
    Ok(())
}

fn pop(stack: &mut Vec<V>) -> Result<V> {
    stack.pop().context("pickle 栈空")
}

fn read_cstr(data: &[u8], i: &mut usize) -> Result<String> {
    let start = *i;
    while *i < data.len() && data[*i] != b'\n' {
        *i += 1;
    }
    if *i >= data.len() {
        bail!("GLOBAL 没有换行");
    }
    let s = String::from_utf8(data[start..*i].to_vec()).context("GLOBAL 不是 utf-8")?;
    *i += 1;
    Ok(s)
}
