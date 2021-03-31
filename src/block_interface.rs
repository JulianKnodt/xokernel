use crate::{
  bit_array::{nearest_div_8, BitArray},
  virtio::Driver,
};
use alloc::prelude::v1::Box;
use core::{any::Any, convert::TryInto};

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
  fn owned(&self) -> &[u32];

  // TODO possibly add associated error types here

  // TODO maybe add a vectorized version of below instead of allocating one at a time which may
  // be costly?

  // TODO these functions need to be purely deterministic, and thus shouldn't contain any unsafe
  // code which allows for raw-ptr manipulation
  // also need to have a functional interface here, to allow for rollback if there are errors.

  fn len(&self) -> usize;

  fn insert(&self, b: u32) -> Result<Self, ()>
  where
    Self: Sized;

  fn remove(&self, b: u32) -> Result<Self, ()>
  where
    Self: Sized;
  /// Serializes this instance into a byte array
  fn ser(&self) -> &[u8];
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

const OWN_BLOCKS: usize = 5;

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

pub struct GlobalBlockInterface<B: BlockDevice>
where
  [(); nearest_div_8(B::NUM_BLOCKS)]: , {
  // TODO maybe condense these fields?
  stored: [Option<Box<dyn Metadata>>; MD_SPACE],
  owners: [Owner; MD_SPACE],

  // There should be a free_map per block-device, but its representation might be generalizable.
  // For now just go with a bit array.
  free_map: BitArray<{ B::NUM_BLOCKS }>,

  // TODO decide whether or not to include a BlockDescriptor sort of thing with an offset?
  // I don't think it's wise to include this, because blocks can't be partially written, it's
  // just all at once.
  block_device: B,
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub struct MetadataHandle(u32);

static mut GLOBAL_BLOCK_INTERFACE: GlobalBlockInterface<Driver> =
  GlobalBlockInterface::new(unsafe { Driver::new() });

pub fn global_block_interface() -> &'static mut GlobalBlockInterface<Driver> {
  unsafe { &mut GLOBAL_BLOCK_INTERFACE }
}

fn as_dyn(md: &Box<dyn Metadata>) -> &dyn Any { md }

impl<B: BlockDevice> GlobalBlockInterface<B>
where
  [(); nearest_div_8(B::NUM_BLOCKS)]: ,
  [(); B::BLOCK_SIZE]: ,
{
  /// Creates an empty instance of a the global block interface
  const fn new(block_device: B) -> Self {
    use core::mem::MaybeUninit;
    let mut stored: [MaybeUninit<Option<Box<dyn Metadata>>>; MD_SPACE] =
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
  pub fn try_init<F>(&mut self, create_metadata: F) -> Result<(), ()>
  where
    F: Fn(&[u8]) -> Result<Box<dyn Metadata>, ()>,
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
      self.stored[i] = Some(create_metadata(&buf[curr..curr + len])?);
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
  pub fn new_metadata<M: Metadata>(&mut self, owner: Owner) -> Result<MetadataHandle, ()> {
    assert_ne!(owner, Owner::NoOwner);
    let md = M::new();
    let starting_owned = md.owned();
    if !starting_owned.is_empty() {
      // We expect metadata to start off empty
      return Err(());
    }
    let free_space = match self.stored.iter_mut().position(|v| v.is_none()) {
      // Out of space
      None => return Err(()),
      Some(space) => space,
    };
    self.stored[free_space] = Some(Box::new(md));
    self.owners[free_space] = owner;
    Ok(MetadataHandle(free_space as u32))
  }

  /// Gets a reference to an instance of Metadata
  pub fn md_ref<M: Metadata>(&mut self, MetadataHandle(i): MetadataHandle) -> Result<&M, ()> {
    let md = self.stored.get(i as usize).ok_or(())?;
    let md = md.as_ref().ok_or(())?;
    let md = as_dyn(&md).downcast_ref::<M>().ok_or(())?;
    Ok(md)
  }

  /// Returns the metadata for a given owner_id, used after rebooting the system.
  pub fn metadatas_for(&self, owner: Owner) -> impl Iterator<Item = MetadataHandle> + '_ {
    self
      .owners
      .iter()
      .enumerate()
      .filter(move |(_, &v)| v == owner)
      .map(|(i, _)| MetadataHandle(i as u32))
  }

  /// Requests an additional block for a specific MetadataHandle. The allocated block is not
  /// guaranteed to be contiguous.
  pub fn req_block<M: Metadata>(
    &mut self,
    MetadataHandle(i): MetadataHandle,
    new_block: u32,
  ) -> Result<(), ()> {
    if self.free_map.get(new_block as usize) {
      return Err(());
    }
    let i = i as usize;
    let md = self.stored.get(i).ok_or(())?;
    let md = md.as_ref().ok_or(())?;
    let mut old_owned = md.owned().iter();

    // let new_b = self.free_map.find_free().ok_or(())? as u32;
    let new_md = as_dyn(&md).downcast_ref::<M>().unwrap().insert(new_block)?;

    let mut new_owned = new_md.owned().iter();
    // They must also maintain order between allocations
    if !new_owned
      .by_ref()
      .zip(old_owned.by_ref())
      .all(|(new, prev)| new == prev)
    {
      return Err(());
    }
    match new_owned.next() {
      Some(&b) if b == new_block => {},
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
    self.stored[i] = Some(Box::new(new_md));

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
    let owned = md.owned();
    let &b_n = owned.get(n).ok_or(())?;

    unsafe { self.block_device.read(b_n, dst) }
  }

  /// Writes to the `n`th block of the metadata handle from src
  pub fn write(
    &self,
    MetadataHandle(i): MetadataHandle,
    n: usize,
    src: &[u8],
  ) -> Result<usize, ()> {
    let i = i as usize;
    let md = self.stored.get(i).ok_or(())?;
    let md = md.as_ref().ok_or(())?;
    let owned = md.owned();
    let &b_n = owned.get(n).ok_or(())?;

    unsafe { self.block_device.write(b_n, src) }
  }
  fn own_required_blocks(&self) -> usize {
    let num_bytes = core::mem::size_of::<u32>()
      + MD_SPACE
      + core::mem::size_of::<BitArray<{ B::NUM_BLOCKS }>>()
      + self
        .stored
        .iter()
        .filter_map(|v| v.as_ref().map(|v| v.len()))
        .sum::<usize>();
    num_bytes / B::BLOCK_SIZE
  }

  /// Persists the state of the global block interface, including allocated blocks, and metadata
  pub fn persist_block_interface(&mut self)
  where
    [(); B::BLOCK_SIZE]: , {
    let num_blocks_required = self.own_required_blocks();
    let mut buf = [0; B::BLOCK_SIZE];
  }
}
