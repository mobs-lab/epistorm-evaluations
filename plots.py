import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

# delete old plots
files = glob.glob('/plots/WIS_ratio/*')
for f in files:
    os.remove(f)

# load data
dfwis_ratio = pd.read_csv('./evaluations/WIS_ratio.csv')

# select 4 most recent dates
dates = sorted(pd.unique(dfwis_ratio.reference_date))[-4:]

# generate plots
for ref_date in dates:

    fig,ax = plt.subplots(2,2,figsize=(20,25))

    for horizon in [0,1,2,3]:

        if horizon ==0:
            i=0
            j=0
        elif horizon ==1:
            i=0
            j=1
        elif horizon == 2:
            i=1
            j=0
        else:
            i=1
            j=1

        # filter by horizon and submission date
        df = dfwis_ratio[(dfwis_ratio.horizon==horizon) & (dfwis_ratio.reference_date == ref_date)].copy()

        if len(df)==0:
            fig.delaxes(ax[i][j])
            continue
        
        # plot vertical line
        ax[i][j].axvline(x=1, ymin=0, ymax=3,linestyle='--', color = 'darksalmon', alpha=1)


        # make boxplots
        my_order = df.groupby(by=['Model'])['wis_ratio'].median().sort_values(ascending=True).index
        g = sns.boxplot(x='wis_ratio', y='Model', data=df,ax = ax[i][j],order=my_order, color = '#17B1BF',width=.4, showfliers=False)

        for patch in ax[i][j].patches:
            r, gr, b, a = patch.get_facecolor()
            patch.set_facecolor((r, gr, b, .1))

        # make underlying distribution of points, swarm plot
       # sns.swarmplot(x='wis_ratio', y='Model', data=df,order=my_order,ax = ax[i][j],color = "#49AFB7",alpha = .5,size=2, orient="h")    

        # formatting
        g.set(ylabel=None)
        g.set(yticklabels=[])
        g.set(yticks=[])

        ax[i][j].set_xlabel('Flusight Forecast model / Flusight-baseline WIS value ', fontsize=13)

        # add text of model name within plot
        a = 0
        for mod in list(my_order):
            if mod =='MOBS-GLEAM_FLUH':
                ax[i][j].text(.1,a-.25, mod, fontsize=12, style='italic', color='dimgray', fontweight='bold')
            else:
                ax[i][j].text(.1,a-.25, mod, fontsize=12, style='italic', color='dimgray')
            
            
            
            a+=1

        # formatting
        ax[i][j].spines["top"].set_visible(False)
        ax[i][j].spines["right"].set_visible(False)
        ax[i][j].spines["left"].set_visible(False)

        ax[i][j].set_title('horizon: ' +str(horizon) , fontsize=15)

    fig.suptitle(f'Forecast Date: {ref_date}',y=1, fontsize=20)

    fig.tight_layout()
    plt.savefig(f'./plots/WIS_{ref_date}.pdf')
