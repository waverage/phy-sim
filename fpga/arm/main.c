/* The ARM side of the board loopback (M8): bare metal on the Zynq's first
 * Cortex-A9, MMU and caches off, loaded by fpga/board.tcl after ps7_init.
 *
 * The PC puts a job in DDR (tools/board.py): for each SNR point, the
 * (word address, data) pairs to write to rtl/loopback (receiver tables,
 * TX frames, information bits, emulator gains and seed) and the number of
 * frames. For each point the ARM writes them, starts the run, waits, and
 * copies the PL's error counters to the results area, where the PC reads
 * them back over JTAG.
 */
#include <stdint.h>

#define PL      ((volatile uint32_t *)0x43C00000u)   /* rtl/loopback, word addresses */
#define JOB     ((volatile uint32_t *)0x10000000u)
#define RESULTS ((volatile uint32_t *)0x0F000000u)

#define JOB_MAGIC 0x424A504Eu   /* "NPJB" */
#define RES_MAGIC 0x52534C54u   /* "TLSR" */

enum { J_MAGIC, J_NPOINTS, J_STATUS, J_POINT, J_POINTS };          /* job header words */
enum { P_OFFSET, P_NWORDS, P_NFRAMES, P_FLAGS, P_WORDS };           /* per point */
enum { LB_CTRL = 0, LB_STATUS = 1, LB_NFRAMES = 2, LB_COUNTERS = 16, N_COUNTERS = 7 };
enum { S_RUNNING = 1, S_DONE = 2, S_BAD_JOB = 0xBAD };

int main(void)
{
    if (JOB[J_MAGIC] != JOB_MAGIC) {
        JOB[J_STATUS] = S_BAD_JOB;
        return 1;
    }
    uint32_t npoints = JOB[J_NPOINTS];
    JOB[J_STATUS] = S_RUNNING;
    for (uint32_t p = 0; p < npoints; p++) {
        volatile uint32_t *pt = &JOB[J_POINTS + P_WORDS * p];
        volatile uint32_t *w = &JOB[pt[P_OFFSET]];
        for (uint32_t i = 0; i < pt[P_NWORDS]; i++)
            PL[w[2 * i]] = w[2 * i + 1];
        PL[LB_NFRAMES] = pt[P_NFRAMES];
        PL[LB_CTRL] = 2;                  /* clear the counters */
        PL[LB_CTRL] = 1;                  /* run */
        while (PL[LB_STATUS] & 1)
            ;
        for (uint32_t k = 0; k < N_COUNTERS; k++)
            RESULTS[8 * p + k] = PL[LB_COUNTERS + k];
        RESULTS[8 * p + 7] = RES_MAGIC;
        JOB[J_POINT] = p + 1;
    }
    JOB[J_STATUS] = S_DONE;
    return 0;
}
