/* Bare-metal entry for fpga/arm/main.c: interrupts off, a stack, a zeroed
 * .bss, main, then wait for the debugger. The MMU and caches stay off
 * (as after reset), so the PL's registers and the job in DDR are uncached. */
    .section .text.boot, "ax"
    .global _start
    .arm
_start:
    cpsid   if
    ldr     sp, =_stack_top
    ldr     r0, =__bss_start
    ldr     r1, =__bss_end
    mov     r2, #0
1:  cmp     r0, r1
    strlo   r2, [r0], #4
    blo     1b
    bl      main
2:  wfe
    b       2b
