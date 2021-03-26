use core::fmt;

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Color {
  Black = 0,
  Blue = 1,
  Green = 2,
  Cyan = 3,
  Red = 4,
  Magenta = 5,
  Brown = 6,
  LightGray = 7,
  DarkGray = 8,
  LightBlue = 9,
  LightGreen = 10,
  LightCyan = 11,
  LightRed = 12,
  Pink = 13,
  Yellow = 14,
  White = 15,
}

#[derive(PartialEq, Eq)]
pub struct Writer {
  row: usize,
  col: usize,
}
impl Writer {
  pub fn new(row: usize, col: usize) -> Self { Self { row, col } }
  pub fn write_byte(&mut self, b: u8) {
    if self.row >= MAX_ROW {
      return;
    }
    if self.col >= MAX_COL {
      return;
    }
    match b {
      b'\n' => {
        self.row += 1;
        self.col = 0;
        return;
      },
      _ => {},
    }
    let p = 2 * (self.row * MAX_COL + self.col) as isize;
    unsafe {
      *VGA_BUFFER.offset(p) = b;
      *VGA_BUFFER.offset(p + 1) = 0xb;
    }
    self.col += 1;
    if self.col == 80 {
      self.col = 0;
      self.row += 1;
    }
  }
}

static mut TEST: usize = 0;
impl fmt::Write for Writer {
  fn write_str(&mut self, s: &str) -> fmt::Result {
    for &b in s.as_bytes() {
      self.write_byte(b);
    }
    Ok(())
  }
}
static mut VGA_BUFFER: *mut u8 = 0xb8000 as *mut u8;

const MAX_ROW: usize = 25;
const MAX_COL: usize = 80;

/// PRINTS
pub(crate) fn print_at(val: &[u8], row: usize, col: usize) {
  if row >= MAX_ROW {
    return;
  }
  if col >= MAX_COL {
    return;
  }
  let start_pos = 2 * (row * MAX_COL + col);
  for (i, &b) in val.iter().take(MAX_ROW * MAX_COL - start_pos).enumerate() {
    let p = 2 * (start_pos + i) as isize;
    unsafe {
      *VGA_BUFFER.offset(p) = b;
      *VGA_BUFFER.offset(p + 1) = 0xb;
    }
  }
}

/// Clears the VGA_BUFFER
pub(crate) fn clear() {
  unsafe {
    core::ptr::write_bytes(VGA_BUFFER, 0, 80 * 25);
  }
}
