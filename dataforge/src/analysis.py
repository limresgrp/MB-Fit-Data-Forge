import numpy as np
import plotly.graph_objects as go
from matplotlib import pyplot as plt


def plot_3d_frame(frame_id, dataset, orig_id=None, show_angles: bool = True):
    pos = dataset.get('R', dataset.get('coords'))
    if orig_id is not None:
        pos = pos[orig_id]
    pos = pos[frame_id]
    bond_idcs = dataset.get('bond_idcs', None)
    if bond_idcs is not None:
        bond_idcs = bond_idcs.T
    angle_idcs = None if not show_angles else dataset.get('angle_idcs', None)
    monomer_names = dataset.get('monomer_names', None)
    if monomer_names is not None:
        text = [f"name: {bt} | monomer: {mn}" for bt, mn in zip(dataset['bead_types'], monomer_names)]
    else:
        text = [f"name: {bt} | id: {i}" for i, bt in enumerate(dataset.get('bead_types', dataset.get('names')))]
    
    color = dataset.get('z', dataset.get('atomic_numbers'))
    trace_atoms = go.Scatter3d(
            x=pos[:, 0],
            y=pos[:, 1],
            z=pos[:, 2],
            name='atoms',
            text=text,
            mode='markers',
            marker=dict(symbol='circle', color=color, size=5, opacity=0.5)
        )
    
    trace_atoms2 = go.Scatter3d(
            x=pos[520:521, 0],
            y=pos[520:521, 1],
            z=pos[520:521, 2],
            name='atoms',
            text=[text[520]],
            mode='markers',
            marker=dict(symbol='circle', color=color, size=30, opacity=0.5)
        )

    data = [trace_atoms, trace_atoms2]

    if bond_idcs is not None:
        x_bonds = []
        y_bonds = []
        z_bonds = []

        for i in range(bond_idcs.shape[1]):
            x_bonds.extend([pos[bond_idcs[0][i], 0].item(), pos[bond_idcs[1][i], 0].item(), None])
            y_bonds.extend([pos[bond_idcs[0][i], 1].item(), pos[bond_idcs[1][i], 1].item(), None])
            z_bonds.extend([pos[bond_idcs[0][i], 2].item(), pos[bond_idcs[1][i], 2].item(), None])

        trace_bonds = go.Scatter3d(
                x=x_bonds,
                y=y_bonds,
                z=z_bonds,
                name='bonds',
                mode='lines',
                line=dict(color='black', width=1),
                hoverinfo='none')
        data.append(trace_bonds)

    if angle_idcs is not None:
        for a_idcs in angle_idcs:
            data.append(
                go.Mesh3d(
                    x=pos[a_idcs, 0],
                    y=pos[a_idcs, 1],
                    z=pos[a_idcs, 2],
                    color='lightpink',
                    opacity=0.50
                )
            )

    layout = go.Layout(
        width=1200,
        height=800,
        plot_bgcolor='rgba(1,1,1,1)',
        paper_bgcolor='rgba(217, 221, 245, 0.25)',
        # xaxis =  {                                     
        #     'showgrid': False
        # },
        # yaxis = {                              
        #     'showgrid': False
        # }
        scene = dict(
            xaxis = dict(
                nticks=3,
                backgroundcolor="rgba(0,0,0,0.2)",
                gridcolor="whitesmoke",
                showbackground=True,
                showgrid=True,
                ),
            yaxis = dict(
                nticks=3,
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor="whitesmoke",
                showbackground=True,
                showgrid=True,
                ),
            zaxis = dict(
                nticks=3,
                backgroundcolor="rgba(0,0,0,0.4)",
                gridcolor="whitesmoke",
                showbackground=True,
                showgrid=True,
                ),
            # bgcolor='rgba(0,0,0,0)',
            ),
    )

    return go.Figure(data=data, layout=layout)

def plot_ramachandran(cv1, cv2, bins=60, energy_levels=50, plot='isolines', edges=None, xticks=None, yticks=None):
    cmap = plt.cm.get_cmap("magma")
    if edges is None:
        edges = np.array([[cv2.min(), cv2.max()], [cv1.min(), cv1.max()]])
    counts, _, _ = np.histogram2d(cv2.reshape(-1),
                                  cv1.reshape(-1),
                                  bins=bins,
                                  range=edges)
    populations = counts / np.sum(counts)
    
    # compute energies for only non-zero entries
    # 1/beta is approximately 0.6 kcal/mol at 300 K
    energies = -0.6*np.log(populations + (1 if np.sum(counts)==1 else 0),
                           out=np.zeros_like(populations),
                           where=(populations > 0))
    
    # make the lowest energy slightly above zero
    min_energies = energies[np.nonzero(energies)]
    if len(min_energies) == 0:
        min_energies = np.zeros((1,), dtype=np.float32)
    energies = np.where(energies,
                        energies-np.min(min_energies) + 1e-6,
                        0)
    
    # mask the zero values from the colormap
    zvals_masked = np.ma.masked_where(energies == 0, energies)

    # bin energies into "energy_levels" bins
    energy_bins = np.linspace(zvals_masked.min(), zvals_masked.max(), energy_levels)
    digitized = np.digitize(zvals_masked, energy_bins)
    for i in range(1, len(energy_bins)):
        bin_mean = zvals_masked[digitized == i].mean()
        zvals_masked[digitized == i] = 0 if np.isnan(bin_mean) else bin_mean

    cmap.set_bad(color='white')
    if plot=='image':
        plt.imshow(zvals_masked, interpolation='nearest', cmap = cmap)
        plt.gca().invert_yaxis()
    elif plot=='isolines':
        try:
            CS = plt.contourf(zvals_masked, levels=energy_bins, cmap = cmap)
            plt.contour(CS, levels=CS.levels[::20], cmap='RdGy', origin='lower')
        except:
            plt.imshow(zvals_masked, interpolation='nearest', cmap = cmap)
            plt.gca().invert_yaxis()
    
    if xticks is None:
        xticks = [
            '{0:.2f}'.format(edges[1,0]),
            '{0:.2f}'.format((edges[1,0] + edges[1,1])/2),
            '{0:.2f}'.format(edges[1,1])
        ]
    plt.xticks(
        [-0.5, bins/2, bins],
        xticks
    )

    if yticks is None:
        yticks = [
            '{0:.2f}'.format(edges[0,0]),
            '{0:.2f}'.format((edges[0,0] + edges[0,1])/2),
            '{0:.2f}'.format(edges[0,1])
        ]
    plt.yticks(
        [-0.5, bins/2, bins],
        yticks
    )
    
    plt.xlabel(r'$TICA 1$',fontsize=16)
    plt.ylabel(r'$TICA 2$',fontsize=16)
    
    cb=plt.colorbar()
    cb.ax.set_title(r'$\tilde{F}\left(\frac{kcal}{mol}\right)$')
    return edges, xticks, yticks