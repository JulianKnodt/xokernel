use crate::bit_array::{nearest_div_8, BitArray};

/*
#[cfg(not(target_os = "macos"))]
use alloc::prelude::v1::Box;
*/

use core::any::Any;

pub trait BlockDevice {
  const NUM_BLOCKS: usize;
  const BLOCK_SIZE: usize;
  /// Read from a block on this device into dst. Returns number of bytes read.
  fn read(&self, block_num: u32, dst: &mut [u8]) -> Result<usize, ()>;
  /// Write to a block on this device from src. Returns number of bytes written.
  fn write(&self, block_num: u32, src: &[u8]) -> Result<usize, ()>;
  /// Perform initialization of this block device
  fn init(&mut self) {}
}

#[macro_export]
macro_rules! default_ser_impl {
  () => {
    /// Serializes this instance into a byte array
    fn ser(&self) -> &[u8] {
      unsafe {
        core::ptr::slice_from_raw_parts(
          (self as *const Self) as *const u8,
          core::mem::size_of::<Self>(),
        )
        .as_ref()
        .unwrap()
      }
    }
  };
}

/// Represents Disk Metadata for some application
pub trait Metadata: 'static {
  fn new() -> Self
  where
    Self: Sized;

  type Iter<'a>: IntoIterator<Item = u32> + 'a;
  fn owned(&self) -> Self::Iter<'_>;

  const LEN: usize;

  // TODO possibly add associated error types here

  // TODO maybe add a vectorized version of below instead of allocating one at a time which may
  // be costly?

  // TODO these functions need to be purely deterministic, and thus shouldn't contain any unsafe
  // code which allows for raw-ptr manipulation
  // also need to have a functional interface here, to allow for rollback if there are errors.

  fn insert(&self, b: u32) -> Result<Self, ()>
  where
    Self: Sized;

  fn remove(&self, b: u32) -> Result<Self, ()>
  where
    Self: Sized;
  /// Serializes this instance into a byte array
  fn ser(&self) -> &[u8]
  where
    Self: Sized;
  /// returns the number of bytes read and an instance of Self
  fn de(bytes: &[u8]) -> Result<(Self, usize), ()>
  where
    Self: Sized, {
    let len = core::mem::size_of::<Self>();
    let used = &bytes.get(..len).ok_or(())?;
    assert!(used.len() == len);
    let v = unsafe { (used.as_ptr() as *const Self).read() };
    Ok((v, len))
  }
}

/// Macro for creating an enum over all possible Metadata, instead of using the dyn approach.
/// This assumption that we know all the metadata may be harder to achieve in practice, but
/// would also give more performance guarantees.
#[macro_export]
macro_rules! mk_metadata_enum {
  ( $( ($md_id: ident, $md: ty)$(,)? )+ ) => {
    /// A specific ID for each metadata registered with the system
    /// This allows for easily deserializing items
    #[repr(u8)]
    #[derive(PartialEq, Eq, Clone, Copy, Debug)]
    pub enum MetadataID {
      $( $md_id, )+
      Unknown
    }

    impl From<u8> for MetadataID {
      fn from(v: u8) -> Self {
        $(
          if v == Self::$md_id as u8 {
            return Self::$md_id
          }
        )+
        return MetadataID::Unknown
      }
    }
    impl MetadataID {
      pub fn len(self) -> usize {
        match self {
          $(Self::$md_id => core::mem::size_of::<$md>(), )+
          Self::Unknown => 0,
        }
      }
    }
    #[derive(Clone, Debug)]
    pub enum AllMetadata {
      $( $md_id($md), )+
    }
    pub enum MetadataIter<'a> {
      $( $md_id(<<$md as Metadata>::Iter<'a> as IntoIterator>::IntoIter), )+
    }

    impl<'a> Iterator for MetadataIter<'a> {
      type Item = u32;
      fn next(&mut self) -> Option<Self::Item> {
        match self {
          $( Self::$md_id(v) => v.next(), )+
        }
      }
    }
    $(
      impl From<$md> for AllMetadata {
        fn from(v: $md) -> Self {
          Self::$md_id(v)
        }
      }
    )+
    impl AllMetadata {
      #[inline]
      pub fn id(&self) -> MetadataID {
        match self {
          $( Self::$md_id(_) => MetadataID::$md_id, )+
        }
      }
      pub fn owned(&self) -> MetadataIter {
        match self {
          $( Self::$md_id(v) => MetadataIter::$md_id(v.owned().into_iter()),  )+
        }
      }
      pub fn insert(&self, b: u32) -> Result<Self, ()> {
        match self {
          $( Self::$md_id(v) => Ok(Self::$md_id(v.insert(b)?)),  )+
        }
      }
      pub fn len(&self) -> usize {
        match self {
          $( Self::$md_id(_) => <$md as Metadata>::LEN,  )+
        }
      }
      pub fn ser(&self) -> &[u8] {
        match self {
          $( Self::$md_id(v) => v.ser(),  )+
        }
      }
      /// Deserializes some bytes into Self
      pub fn de(id: MetadataID, bytes: &[u8]) -> Result<Self, ()> {
        match id {
          $( MetadataID::$md_id => {
            let (v, read) = <$md>::de(bytes)?;
            assert_eq!(read, bytes.len());
            Ok(v.into())
          }  )+
          MetadataID::Unknown => Err(()),
        }
      }
    }
  }
}

mk_metadata_enum!(
  (Superblock, crate::fs::Superblock),
  (RangeMetadata, crate::fs::RangeMetadata),
);

const OWN_BLOCKS: usize = 5;

/// What is the first block that can be used by LibFS's
pub const FIRST_FREE_BLOCK: u32 = 5;

// Number of Metadata items stored in this block interface.
const MD_SPACE: usize = 32;

const MAGIC_NUMBER: u32 = 0xdea1d00d;

#[repr(u8)]
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum Owner {
  /// Owned by the kernel for persisting its own metadata
  Ours = 0,
  LibFS = 1,
  /// Metadata currently has no owner
  NoOwner = 255,
}
impl Owner {
  fn from_byte(b: u8) -> Self {
    match b {
      0 => Self::Ours,
      1 => Self::LibFS,
      255 => Self::NoOwner,
      _ => panic!("Unknown owner byte"),
    }
  }
}

/// The struct for a singleton which interfaces with the device on behalf of LibFSs.
#[derive(Debug)]
pub struct GlobalBlockInterface<B: BlockDevice>
where
  [(); nearest_div_8(B::NUM_BLOCKS)]: , {
  stored: [Option<AllMetadata>; MD_SPACE],
  owners: [Owner; MD_SPACE],

  // There should be a free_map per block-device,
  // but its representation might be generalizable. For now just go with a bit array.
  free_map: BitArray<{ B::NUM_BLOCKS }>,

  block_device: B,
}

/// A capability for metadata.
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub struct MetadataHandle(u32);

impl<B: BlockDevice> GlobalBlockInterface<B>
where
  [(); nearest_div_8(B::NUM_BLOCKS)]: ,
  [(); B::BLOCK_SIZE]: ,
{
  /// Creates an empty instance of a the global block interface
  pub const fn new(block_device: B) -> Self {
    use core::mem::MaybeUninit;
    let mut stored: [MaybeUninit<Option<AllMetadata>>; MD_SPACE] =
      MaybeUninit::uninit_array::<MD_SPACE>();
    let mut i = 0;
    // necessary to use a while loop here because of constraints on const fns.
    while i < MD_SPACE {
      stored[i].write(None);
      i += 1;
    }
    let stored: [Option<_>; MD_SPACE] = unsafe { core::mem::transmute(stored) };
    let mut free_map = BitArray::new(false);
    // This is because of const fns again
    let mut i = 0;
    while i < OWN_BLOCKS {
      free_map.set(i);
      i += 1;
    }
    Self {
      stored,
      owners: [Owner::NoOwner; MD_SPACE],
      free_map,
      block_device,
    }
  }

  /// Tries to initialize this
  pub fn try_init(&mut self) -> Result<(), ()>
  where
    [(); OWN_BLOCKS * B::BLOCK_SIZE]: , {
    self.block_device.init();

    let mut buf = [0; OWN_BLOCKS * B::BLOCK_SIZE];
    for i in 0..OWN_BLOCKS {
      let read = self
        .block_device
        .read(i as u32, &mut buf[i * B::BLOCK_SIZE..])?;
      assert_eq!(read, B::BLOCK_SIZE);
    }
    // --- read magic number
    if u32::from_ne_bytes([buf[0], buf[1], buf[2], buf[3]]) != MAGIC_NUMBER {
      // Never previously serialized this item, nothing more to do
      return self.persist();
    }
    // --- read free map
    let mut curr = 4;
    let len = self.free_map.items.len();
    self.free_map.items.copy_from_slice(&buf[curr..curr + len]);
    curr += len;
    // --- read metadata: [OWNER(u8); LEN(u32); DATA(&[u8])],
    // if OWNER::NoOwner skip next fields
    for i in 0..MD_SPACE {
      let owner = Owner::from_byte(buf[curr]);
      curr += 1;
      self.owners[i] = owner;
      if owner == Owner::NoOwner {
        continue;
      }
      let len =
        u32::from_ne_bytes([buf[curr], buf[curr + 1], buf[curr + 2], buf[curr + 3]]) as usize;
      curr += 4;
      let id = MetadataID::from(buf[curr]);
      curr += 1;
      let len = id.len();
      self.stored[i] = Some(AllMetadata::de(id, &buf[curr..curr +len])?);
      curr += len;
    }
    Ok(())
  }

  pub fn persist(&self) -> Result<(), ()>
  where
    [(); OWN_BLOCKS * B::BLOCK_SIZE]: , {
    let mut buf = [0; OWN_BLOCKS * B::BLOCK_SIZE];
    // --- write magic number
    let bytes = u32::to_ne_bytes(MAGIC_NUMBER);
    buf[..4].copy_from_slice(&bytes);
    // --- write free map
    let mut curr = 4;
    let len = self.free_map.items.len();
    buf[curr..curr + len].copy_from_slice(&self.free_map.items);
    curr += len;
    // --- read metadata: [OWNER(u8); LEN(u32); DATA(&[u8])],
    // if OWNER::NoOwner skip next fields
    for i in 0..MD_SPACE {
      let owner = self.owners[i];
      buf[curr] = owner as u8;
      curr += 1;
      if owner == Owner::NoOwner {
        continue;
      }
      let md = self.stored[i]
        .as_ref()
        .expect("No metadata but owner")
        .ser();
      let len = md.len();
      buf[curr..curr + 4].copy_from_slice(&u32::to_ne_bytes(len as u32));
      curr += 4;
      buf[curr..curr + len].copy_from_slice(md);
      curr += len;
    }
    for i in 0..OWN_BLOCKS {
      let written = self
        .block_device
        .write(i as u32, &mut buf[i * B::BLOCK_SIZE..])?;
      assert_eq!(written, B::BLOCK_SIZE);
    }
    Ok(())
  }

  pub fn free_map(&self) -> &BitArray<{ B::NUM_BLOCKS }> { &self.free_map }

  /// Allocates new metadata inside of the system
  pub fn new_metadata<M: Metadata + Into<AllMetadata>>(
    &mut self,
    owner: Owner,
  ) -> Result<MetadataHandle, ()> {
    assert_ne!(owner, Owner::NoOwner);
    let md = M::new();
    let starting_owned = md.owned();
    if starting_owned.into_iter().next().is_some() {
      // We expect metadata to start off empty
      return Err(());
    }
    let free_space = match self.stored.iter_mut().position(|v| v.is_none()) {
      // Out of space
      None => return Err(()),
      Some(space) => space,
    };
    self.stored[free_space] = Some(md.into());
    self.owners[free_space] = owner;
    Ok(MetadataHandle(free_space as u32))
  }

  /// Gets a reference to an instance of Metadata
  pub fn md_ref(&mut self, MetadataHandle(i): MetadataHandle) -> Result<&AllMetadata, ()> {
    let md = self.stored.get(i as usize).ok_or(())?;
    let md = md.as_ref().ok_or(())?;
    Ok(md)
  }

  /// Returns the metadata for a given owner_id, used after rebooting the system.
  pub fn metadatas_for(
    &self,
    owner: Owner,
  ) -> impl Iterator<Item = (MetadataHandle, &'_ AllMetadata)> + '_ {
    self
      .owners
      .iter()
      .enumerate()
      .filter(move |(_, &v)| v == owner)
      .filter_map(move |(i, _)| {
        self.stored[i]
          .as_ref()
          .map(|amd| (MetadataHandle(i as u32), amd))
      })
  }

  /// Requests an additional block for a specific MetadataHandle. The allocated block is not
  /// guaranteed to be contiguous.
  pub fn req_block(&mut self, MetadataHandle(i): MetadataHandle, new_block: u32) -> Result<(), ()> {
    if self.free_map.get(new_block as usize) {
      return Err(());
    }
    let i = i as usize;
    let md = self.stored.get(i).ok_or(())?;
    let md = md.as_ref().ok_or(())?;
    let mut old_owned = md.owned();

    // let new_b = self.free_map.find_free().ok_or(())? as u32;
    let new_md = md.insert(new_block)?;

    let mut new_owned = new_md.owned();
    // They must also maintain order between allocations
    if !new_owned
      .by_ref()
      .zip(old_owned.by_ref())
      .all(|(new, prev)| new == prev)
    {
      return Err(());
    }
    match new_owned.next() {
      Some(b) if b == new_block => {},
      // inserted wrong block
      Some(_) => return Err(()),
      // no block seems to have been inserted
      None => return Err(()),
    };

    if old_owned.next().is_some() {
      // Some block was removed and not kept in the new metadata
      return Err(());
    }
    // perform updates after all checks
    self.free_map.set(new_block as usize);
    self.stored[i] = Some(new_md);

    Ok(())
  }

  /// Reads from `n`th block of the metadata handle into dst
  pub fn read(
    &self,
    MetadataHandle(i): MetadataHandle,
    n: usize,
    dst: &mut [u8],
  ) -> Result<usize, ()> {
    let i = i as usize;
    let md = self.stored.get(i).ok_or(())?;
    let md = md.as_ref().ok_or(())?;
    let b_n = md.owned().nth(n).ok_or(())?;

    self.block_device.read(b_n, dst)
  }

  /// Writes to the `n`th block of the metadata handle from src
  pub fn write(
    &self,
    MetadataHandle(i): MetadataHandle,
    n: usize,
    src: &[u8],
  ) -> Result<usize, ()> {
    let i = i as usize;
    let md = &self.stored.get(i).ok_or(())?;
    let md = md.as_ref().ok_or(())?;
    let b_n = md.owned().nth(n).ok_or(())?;

    self.block_device.write(b_n, src)
  }
  #[allow(dead_code)]
  fn own_required_blocks(&self) -> usize {
    let num_bytes = core::mem::size_of::<u32>()
      + MD_SPACE
      + core::mem::size_of::<BitArray<{ B::NUM_BLOCKS }>>()
      + self
        .stored
        .iter()
        .filter_map(|v| Some(v.as_ref()?.len()))
        .sum::<usize>();
    num_bytes / B::BLOCK_SIZE
  }

  /// Persists the state of the global block interface, including allocated blocks, and metadata
  pub fn persist_block_interface(&mut self)
  where
    [(); B::BLOCK_SIZE]: , {
    // let num_blocks_required = self.own_required_blocks();
    //let mut buf = [0; B::BLOCK_SIZE];
    todo!()
  }
}
