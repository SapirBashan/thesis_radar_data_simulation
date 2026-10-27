import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_results(rows,radar,example,output):
    groups={}
    for row in rows:
        key=tuple(row[k] for k in ('format','snr_db','rcs','target_pfa'))
        groups.setdefault(key,[]).append(row)
    for key,points in groups.items():
        fmt,snr,rcs,pfa=key
        fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
        for index,r in enumerate(points):
            color=f'C{index%10}'
            exact=np.isinf(r['mean_sqnr_db'])
            # Exact reconstructions use a separate labelled row, not a fake dB value.
            if not exact:
                axes[0].scatter(r['compression_ratio'],r['mean_sqnr_db'],color=color,label=r['codec'])
            if r['pfa_control_status']=='within_tolerance':
                axes[1].errorbar(r['compression_ratio'],r['pd'],yerr=[[r['pd']-r['pd_ci95_low']],[r['pd_ci95_high']-r['pd']]],fmt='o',color=color,
                                 label=f"{r['codec']} (Pfa={r['achieved_pfa']:.3g})")
        exact_names='; '.join(f"{r['codec']} @ {r['compression_ratio']:.3g}×" for r in points if np.isinf(r['mean_sqnr_db']))
        axes[0].set(title='Reconstruction quality',xlabel='Raw sample bytes / stored bytes',ylabel='Mean per-frame SQNR (dB)')
        axes[0].text(.02,.02,'Exact (SQNR=∞):\n'+exact_names,transform=axes[0].transAxes,fontsize=7,wrap=True,
                     bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
        axes[0].legend(fontsize=7,loc='upper right')
        axes[1].set(title=f'Pd at empirically checked Pfa≈{pfa:g}',xlabel='Raw sample bytes / stored bytes',ylabel='Pd (95% Wilson interval)',ylim=(-.05,1.05))
        if not any(r['pfa_control_status']=='within_tolerance' for r in points):
            axes[1].text(.5,.5,'No methods passed exposure/tolerance checks.\nIncrease null/calibration trials.',transform=axes[1].transAxes,ha='center')
        else:
            axes[1].legend(fontsize=7,loc='upper right')
        for ax in axes: ax.grid(alpha=.25)
        fig.suptitle(f'{fmt}; reference SNR={snr:g} dB; RCS={rcs:g}')
        fig.savefig(output/f'curves_{fmt}_snr{snr}_rcs{rcs}_pfa{pfa}.png',dpi=150)
        plt.close(fig)
    if example:
        original,restored,fmt,name=example
        maps=[radar.features(x)[2] for x in (original,restored)]
        maps_db=[10*np.log10(np.maximum(m,1e-30)) for m in maps]
        lo,hi=np.percentile(np.concatenate([m.ravel() for m in maps_db]),[1,99.9])
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        dr=299792458/(2*radar.s.fs)
        frequencies=np.fft.fftshift(np.fft.fftfreq(radar.s.pulses,1/radar.s.prf))
        df=radar.s.prf/radar.s.pulses
        for ax,m,title in zip(axes,maps_db,('Original reference',f'Reconstructed {name}')):
            image=ax.imshow(m.T,origin='lower',aspect='auto',vmin=lo,vmax=hi,
                            extent=[-dr/2,(radar.s.range_gates-.5)*dr,frequencies[0]-df/2,frequencies[-1]+df/2])
            ax.set(title=title,xlabel='Matched-filter range coordinate (m)',ylabel='Doppler (Hz)')
        fig.colorbar(image,ax=axes,label='Unnormalized FFT power (dB, arbitrary reference)')
        fig.suptitle(fmt+'; shared color limits')
        fig.savefig(output/'range_doppler_example.png',dpi=150)
        plt.close(fig)
