import numpy as np, ast, re, pickle, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D

# ── Load pion data ─────────────────────────────────────────────────────────────
with open('3_-_ALICE_ratios_plot.txt') as fh:
    txt = fh.read()

def parse_sec(txt, n):
    s0=txt.find(str(n)+')'); s1=txt.find(str(n+1)+')',s0) if n<5 else len(txt)
    sec=txt[s0:s1]
    raw=sec[:sec.find('#### RATIOS PLOT') if '#### RATIOS PLOT' in sec else len(sec)]
    def gv(t,v):
        m=re.search(v+'=(\\[.*?\\])\\n',t,re.DOTALL); return ast.literal_eval(m.group(1)) if m else None
    cm2=re.search('centrality_classes=(\\[.*?\\])',sec)
    return {'cent':ast.literal_eval(cm2.group(1)) if cm2 else [],
            'xT':[np.array(x) for x in (gv(raw,'xT') or [])],
            'U': [np.array(u) for u in (gv(raw,'U_xT') or [])],
            'Ue':[np.array(e) for e in (gv(raw,'U_xT_uncertainties') or [])]}

pi_pb = parse_sec(txt, 1)
pi_xe = parse_sec(txt, 3)

with open('/home/claude/data_kp.pkl','rb') as fh:
    kp = pickle.load(fh)

# ── Color palettes ────────────────────────────────────────────────────────────
n_pb = 10; n_xe = len(pi_xe['cent'])  # 9
pal_pb = [cm.Blues_r(0.15 + 0.70*i/(n_pb-1)) for i in range(n_pb)]
pal_xe = [cm.Oranges_r(0.15 + 0.70*i/(n_xe-1)) for i in range(n_xe)]
mrk = ['o','s','^','D','v','p','P','*','h','X']

plt.rcParams.update({'font.family':'DejaVu Serif','font.size':11,
                     'axes.linewidth':0.8,'xtick.direction':'in',
                     'ytick.direction':'in','xtick.top':True,'ytick.right':True,
                     'xtick.minor.visible':True,'ytick.minor.visible':True})

def draw_data(ax, xT_list, U_list, Ue_list, cols, cent_list, filled=True, ms=3):
    for i,(xT,U,Ue,c) in enumerate(zip(xT_list,U_list,Ue_list,cent_list)):
        kw = dict(markersize=ms, elinewidth=0.5, capsize=0, alpha=0.88,
                  color=cols[i], fmt=mrk[i%len(mrk)])
        if not filled:
            kw['markerfacecolor']='none'; kw['markeredgewidth']=0.8
        ax.errorbar(xT, U, yerr=Ue, **kw)

def style_ax(ax, title):
    ax.set_xscale('log')
    ax.set_xlim(0.12, 8)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r'$x_T = p_T\,/\,\langle p_T\rangle$', fontsize=12)
    ax.set_title(title, fontsize=13, pad=5)
    ax.grid(True, which='major', alpha=0.12, lw=0.5, color='gray')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda x,_: {0.2:'0.2',0.5:'0.5',1:'1',2:'2',5:'5'}.get(round(x,1),'')))
    ax.text(0.97,0.96,'ALICE',transform=ax.transAxes,ha='right',va='top',
            fontsize=11,style='italic',fontweight='bold')

fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), sharey=True)
fig.subplots_adjust(wspace=0.07, left=0.058, right=0.992, top=0.87, bottom=0.13)

# ─── Panel 1: Pions ──────────────────────────────────────────────────────────
ax = axes[0]
draw_data(ax, pi_pb['xT'], pi_pb['U'], pi_pb['Ue'], pal_pb, pi_pb['cent'], filled=True)
draw_data(ax, pi_xe['xT'], pi_xe['U'], pi_xe['Ue'], pal_xe, pi_xe['cent'], filled=False)
style_ax(ax, r'$\pi^{\pm}$')
ax.set_ylabel(r'$U(x_T)$', fontsize=14)

# ─── Panel 2: Kaons ──────────────────────────────────────────────────────────
ax = axes[1]
draw_data(ax, kp['xT_K_pb'], kp['U_K_pb'], kp['Uerr_K_pb'], pal_pb, kp['cent_pb'], filled=True)
style_ax(ax, r'$K^{\pm}$')

# ─── Panel 3: Protons ─────────────────────────────────────────────────────────
ax = axes[2]
draw_data(ax, kp['xT_p_pb'], kp['U_p_pb'], kp['Uerr_p_pb'], pal_pb, kp['cent_pb'], filled=True)
style_ax(ax, r'$p + \bar{p}$')

# ─── Legends ──────────────────────────────────────────────────────────────────
cent_labels = [c+'%' for c in pi_pb['cent']]
h_pb = [Line2D([0],[0],color=pal_pb[i],marker=mrk[i%len(mrk)],markersize=5,ls='',
               label=cent_labels[i]) for i in range(n_pb)]
h_xe = [Line2D([0],[0],color=pal_xe[i],marker=mrk[i%len(mrk)],markersize=5,ls='',
               mfc='none',mew=0.9,label=pi_xe['cent'][i]+'%') for i in range(n_xe)]

# Pion panel: two legend boxes (PbPb and XeXe)
lg1=axes[0].legend(handles=h_pb[:5], loc='lower left', fontsize=8, ncol=1,
                   framealpha=0.85, title='PbPb 2.76 TeV', title_fontsize=8.5,
                   edgecolor='0.7', bbox_to_anchor=(0.0,0.0))
axes[0].add_artist(lg1)
lg2=axes[0].legend(handles=h_pb[5:], loc='lower left', fontsize=8, ncol=1,
                   framealpha=0.85, edgecolor='0.7', bbox_to_anchor=(0.22,0.0))
axes[0].add_artist(lg2)
lg3=axes[0].legend(handles=h_xe[:5], loc='lower left', fontsize=8, ncol=1,
                   framealpha=0.85, title='XeXe 5.44 TeV', title_fontsize=8.5,
                   edgecolor='0.7', bbox_to_anchor=(0.44,0.0))
axes[0].add_artist(lg3)
lg4=axes[0].legend(handles=h_xe[5:], loc='lower left', fontsize=8, ncol=1,
                   framealpha=0.85, edgecolor='0.7', bbox_to_anchor=(0.66,0.0))
axes[0].add_artist(lg4)

# Kaon & proton panels: PbPb legend only
for ax in [axes[1], axes[2]]:
    lga=ax.legend(handles=h_pb[:5], loc='lower left', fontsize=8, ncol=1,
                  framealpha=0.85, title='PbPb 2.76 TeV', title_fontsize=8.5,
                  edgecolor='0.7', bbox_to_anchor=(0.0,0.0))
    ax.add_artist(lga)
    lgb=ax.legend(handles=h_pb[5:], loc='lower left', fontsize=8, ncol=1,
                  framealpha=0.85, edgecolor='0.7', bbox_to_anchor=(0.22,0.0))
    ax.add_artist(lgb)

fig.suptitle(r'$U(x_T) = \frac{\langle p_T\rangle}{N}\frac{dN}{dp_T}$   '
             r'($x_T = p_T/\langle p_T\rangle$)   —   Pb–Pb 2.76 TeV & Xe–Xe 5.44 TeV   ALICE',
             fontsize=12, y=0.98)

plt.savefig('/mnt/user-data/outputs/UxT_species_final.png', dpi=200, bbox_inches='tight')
plt.savefig('/mnt/user-data/outputs/UxT_species_final.pdf', bbox_inches='tight')
print("Done.")
