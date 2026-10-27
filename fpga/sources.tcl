# The board designs' RTL, relative to rtl/ (sourced by the other scripts):
# rtl/loopback (M8) and rtl/onchip (M9) share the receiver.
set RTL_SOURCES {
    loopback/loopback_axi.v loopback/loopback.v
    onchip/onchip_axi.v onchip/onchip.v tx/tx_frame.v sync/rx_sync.v stf_corr/stf_corr.v nco_rot/nco_rot.v
    chan_emu/chan_emu.v chan_emu/chan_emu_rtab.v
    rx/rx_frame.v rx/rx_fe.v rx/rx_chest.v rx/rx_track.v rx/rx_demap.v
    fft256/fft256.v fft256/fft256_twiddle.v cordic/cordic.v ldpc_dec/ldpc_dec.v
}
set PART xc7z020clg400-1
