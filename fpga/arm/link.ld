/* fpga/arm: the program in DDR at 1 MB (below the job at 0x10000000 and
 * the results at 0x0f000000), with a 64 KB stack. */
ENTRY(_start)
MEMORY { DDR (rwx) : ORIGIN = 0x00100000, LENGTH = 0x00100000 }
SECTIONS
{
    .text : { *(.text.boot) *(.text*) *(.rodata*) } > DDR
    .data : { *(.data*) } > DDR
    .bss (NOLOAD) : { __bss_start = .; *(.bss*) *(COMMON) . = ALIGN(4); __bss_end = .; } > DDR
    . = ALIGN(8);
    . = . + 0x10000;
    _stack_top = .;
}
