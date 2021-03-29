pub const fn nearest_div_8(v: usize) -> usize { v / 8 + if v % 8 == 0 { 0 } else { 1 } }

#[derive(Debug, PartialEq, Eq)]
pub struct BitArray<const N: usize>
where
  [(); nearest_div_8(N)]: , {
  items: [u8; nearest_div_8(N)],
}

impl<const N: usize> BitArray<N>
where
  [(); nearest_div_8(N)]: ,
{
  pub const fn new(full: bool) -> Self {
    let token = if full { 0xff } else { 0 };
    BitArray {
      items: [token; nearest_div_8(N)],
    }
  }
  /// Sets bit `i` in this bit array.
  pub fn set(&mut self, i: usize) {
    assert!(i < N);
    self.items[i / 8] |= (1 << (i % 8));
  }
  /// Unsets bit `i` in this bit array.
  pub fn unset(&mut self, i: usize) {
    assert!(i < N);
    self.items[i / 8] &= !(1 << (i % 8));
  }
  /// Gets bit `i` in this bit array, where 1 => true, 0 => false.
  pub fn get(&self, i: u32) -> bool {
    let i = i as usize;
    assert!(i < N);
    ((self.items[i / 8] >> (i % 8)) & 1) == 1
  }
  /// Iterates through this bit array trying to find a free block, returning none if there are
  /// none.
  pub fn find_free(&self) -> Option<usize> {
    self
      .items
      .iter()
      .take(N - 1)
      .enumerate()
      .find(|(_, &v)| v != 0xff)
      .map(|(i, v)| {
        // TODO is this leading or trailing ones?
        i * 8 + v.trailing_ones() as usize
      })
      .or_else(|| {
        // need to handle last item which may not be full byte
        let rem = N % 8;
        let mask = (0xFF << rem) >> rem;
        let last = self.items.last().unwrap() & mask;
        if last == mask {
          return None;
        }
        Some(8 * (N - 1) + last.trailing_ones() as usize)
      })
  }
}
