#[macro_export]
macro_rules! cli {
  () => { unsafe { asm!("cli"); } };
  (fl) => {{
    let o: u64;
    asm!("pushfl", "popl {0}", "cli", out(reg) o);
    o
  }}
}
