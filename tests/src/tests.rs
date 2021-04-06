use crate::{block_interface::*, fs, linux_files};

macro_rules! fs_test {
  ($test_name: ident, |$fs_name: ident| $contents: block) => {
    #[test]
    fn $test_name() {
      static mut GBI: GlobalBlockInterface<linux_files::Driver> =
        GlobalBlockInterface::new(linux_files::Driver::new(stringify!($test_name)));
      let mut $fs_name = unsafe {
        GBI.try_init().expect("Failed to init interface");
        fs::FileSystem::new(&mut GBI)
      };
      $contents($fs_name);
      unsafe {
        GBI.block_device.clean();
      }
    }
  };
}

fs_test!(do_init, |gfs| {
  gfs
    .root_dir(fs::FileMode::R)
    .expect("failed to open root dir");
});

fs_test!(add_files_to_root_dir, |gfs| {
  let root_dir_fd = gfs
    .root_dir(fs::FileMode::R)
    .expect("failed to open root dir");
  for i in 0..50 {
    gfs
      .open(root_dir_fd, &[&i.to_string()], fs::FileMode::RW)
      .expect(&format!("Failed to create {}", i));
  }
});
