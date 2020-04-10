# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 12:47:27 2019

@authors: Alvar Escriva-Bou & Rui Hui
If you have any questions, send an email to: alesbou@gmail.com

Python 2.7

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams, ticker
import seaborn as sns
from scipy.stats import ttest_rel
from scipy.stats import levene
from scipy.stats import shapiro
from scipy.stats import norm

#Working inside subdirectory
abspath = os.path.abspath(__file__)
absname = os.path.dirname(abspath)
os.chdir(absname)

#Read original data
dataregions = pd.read_csv('dataregions.csv')
dataprecip = pd.read_csv('dataprecip.csv')
datadroughts = pd.read_csv('datadroughts.csv')
dataspecyield = pd.read_csv('regionspecificyield.csv')
datashadow = pd.read_csv('datashadow.csv')

#dataregions to international units
dataregions['surfdiv'] = dataregions.surfdiv * 1.233480
dataregions['pumping'] = dataregions.pumping * 1.233480
dataregions['et'] = dataregions.et * 1.233480
dataregions['gwstor'] = dataregions.gwstor * 1.233480

#Creating additional datasets from original data
datacv = dataregions.groupby(['model','wateryear']).sum().reset_index()
datatulare = dataregions[dataregions.region>13].groupby(['model', 'wateryear']).sum().reset_index()
data18 = dataregions[dataregions.region==18].groupby(['model', 'wateryear']).sum().reset_index()
dataprecip7503 = dataprecip[(dataprecip.wateryear>1974) & (dataprecip.wateryear<2004)]
datadroughts = datadroughts[datadroughts.wateryear<2004]
datasac = dataregions[dataregions.region<8]
datasj = dataregions[(dataregions.region>7) & (dataregions.region<14)]
datatul = dataregions[dataregions.region>13]
databasins = dataregions
databasins['basin'] = 'Sac'
databasins.basin[(databasins.region>7) & (databasins.region<10)] = 'Del'
databasins.basin[(databasins.region>9) & (databasins.region<14)] = 'SJR'
databasins.basin[(databasins.region>13)] = 'Tul'
databasins = databasins.groupby(['wateryear','basin', 'model']).sum().reset_index()

regioname = ['Sacramento River above Red Bluff','Red Bluff to Chico Landing','Colusa Trough','Chico Landing to Knights Landing','Eastern Sacramento Valley foothills near Sutter Buttes','Cache-Putah area','East of Feather and south of Yuba Rivers','Valley floor east of the Delta','Delta','Delta-Mendota Basin','Modesto and southern Eastern San Joaquin Basin','Turlock Basin','Merced, Chowchilla, and Madera Basins','Westside and Northern Pleasant Valley Basins','Tulare Lake and Western Kings Basin','Northern Kings Basin ','Southern Kings Basin','Kaweah and Tule Basins','Western Kern County and Southern Pleasant  Valley','Northeastern Kern County Basin','Southeastern Kern County Basin']
        

#First we initialize parameters for all figures
params = {
   'axes.labelsize': 8,
   'font.size': 8,
   'legend.fontsize': 8,
   'xtick.labelsize': 8,
   'ytick.labelsize': 8,
   'text.usetex': False,
   'figure.figsize': [6.5, 9.5]
   }
rcParams.update(params)

# =============================================================================
# Figure SI.1: Economic cost from shadow values
# =============================================================================
dataeconcost = pd.DataFrame()
for region in np.unique(datashadow.region):
    datashadowregion = datashadow[datashadow.region == region]
    datashadowregion = datashadowregion.reset_index()
    for j in np.arange(0,1):
        datashadowregion['econcost'] = 0.5*(datashadowregion.shortage[j])*(datashadowregion.shadow[j])
        cumcost = datashadowregion.econcost[j]
    for j in np.arange(1,len(datashadowregion)):
        datashadowregion.econcost[j] = (datashadowregion.shortage[j] - datashadowregion.shortage[j-1])*(datashadowregion.shadow[j-1]+0.5*((datashadowregion.shadow[j]-datashadowregion.shadow[j-1])/2)) + cumcost
        cumcost = datashadowregion.econcost[j]
    if region == np.unique(datashadow.region)[0]:
        dataeconcost = datashadowregion
    else:
        dataeconcost = pd.concat([dataeconcost, datashadowregion])

dataeconcost['shortagemcm'] = dataeconcost.shortage * 1.233480

#Plotting    
sacramentoregions = [1,2,3,4,5,6,7]
sanjoaquinregions = [8,9,10,11,12,13]
tulareregions = [14,15,16,17,18,19,20,21]

fig3 = plt.figure(0, figsize=(6.5,13))
ax = fig3.add_subplot(3, 1, 1)
for region in sacramentoregions:
    ax.plot(0.001*dataeconcost.shortagemcm[dataeconcost.region==region],0.000001*dataeconcost.econcost[dataeconcost.region==region])
    ax.text(0.001*dataeconcost.shortagemcm[dataeconcost.region==region].max(), 0.000001*dataeconcost.econcost[dataeconcost.region==region].max() + 5, region)
ax.set_frame_on(False)
ax.set_xlabel('Reduction in annual net water use (millions of cubic meters)')
ax.set_ylabel('Economic cost (millions of $)')
ax.set_title('Sacramento Valley sub-regions')
ax.set_xlim([0,2000])
ax.set_ylim([0,500])

ax = fig3.add_subplot(3, 1, 2)
for region in sanjoaquinregions:
    ax.plot(0.001*dataeconcost.shortagemcm[dataeconcost.region==region],0.000001*dataeconcost.econcost[dataeconcost.region==region])
    ax.text(0.001*dataeconcost.shortagemcm[dataeconcost.region==region].max(), 0.000001*dataeconcost.econcost[dataeconcost.region==region].max() + 5, region)
ax.set_frame_on(False)
ax.set_xlabel('Reduction in annual net water use (millions of cubic meters)')
ax.set_ylabel('Economic cost (millions of $)')
ax.set_title('Delta, East Side Streams and San Joaquin sub-regions')
ax.set_xlim([0,2000])
ax.set_ylim([0,500])

ax = fig3.add_subplot(3, 1, 3)
for region in tulareregions:
    ax.plot(0.001*dataeconcost.shortagemcm[dataeconcost.region==region],0.000001*dataeconcost.econcost[dataeconcost.region==region])
    ax.text(0.001*dataeconcost.shortagemcm[dataeconcost.region==region].max(), 0.000001*dataeconcost.econcost[dataeconcost.region==region].max() + 5, region)
ax.set_frame_on(False)
ax.set_xlabel('Reduction in annual net water use (millions of cubic meters)')
ax.set_ylabel('Economic cost (millions of $)')
ax.set_title('Tulare Lake sub-regions')
ax.set_xlim([0,2000])
ax.set_ylim([0,500])
plt.savefig('FigSI1.pdf',bbox_inches='tight')



# =============================================================================
# Figure 4: Precipitation, surface diversions, and change in groundwater storage
# =============================================================================
fig4, axs = plt.subplots(3,1)

#converting precip to mm
dataprecip7503['precipstatewidemm'] = dataprecip7503.precipstatewide * 25.4 

axs[0].bar(datadroughts.wateryear, datadroughts.drought, 1.01, color = 'lightgray', label='Drought')
axs[0].bar(dataprecip7503.wateryear, dataprecip7503.precipstatewidemm, label='Precipitation', color='dodgerblue')
axs[0].set_ylim([0,1400])
axs[0].set_ylabel('Statewide average precipitation (mm)')
axs[0].yaxis.set_label_coords(-0.125,0.5)

bar_width = 0.4
axs[1].bar(datadroughts.wateryear, datadroughts.drought, 1.01, color = 'lightgray', label='Drougth')
axs[1].bar(datacv.wateryear[datacv.model == 'cvhm'] - bar_width/2 - 0.02,  datacv.surfdiv[datacv.model == 'cvhm'], bar_width, label='CVHM', color = 'darkorange')
axs[1].bar(datacv.wateryear[datacv.model == 'c2vsim'] + bar_width/2 + 0.02,  datacv.surfdiv[datacv.model == 'c2vsim'] , bar_width, label = 'C2VSim', color = 'mediumpurple')
axs[1].set_ylim([0,25000])
axs[1].set_ylabel('Total surface diversions\nin the Central Valley\n(millions of cubic meters)')
axs[1].yaxis.set_label_coords(-0.1,0.5)

axs[2].bar(datadroughts.wateryear, datadroughts.drought, 1.01, color = 'lightgray', label = 'Drought')
axs[2].bar(datadroughts.wateryear, -datadroughts.drought, 1.01, color = 'lightgray')
axs[2].bar(datacv.wateryear[datacv.model == 'cvhm']- bar_width/2 - 0.02,  datacv.gwstor[datacv.model == 'cvhm'], bar_width, label='CVHM', color = 'darkorange')
axs[2].bar(datacv.wateryear[datacv.model == 'c2vsim'] + bar_width/2 + 0.02,  datacv.gwstor[datacv.model == 'c2vsim'], bar_width , label = 'C2VSim', color = 'mediumpurple')
axs[2].set_ylim([-20000,20000])
axs[2].set_ylabel('Change in annual groundwater\nstorage in the Central Valley\n(millions of cubic meters)')
axs[2].yaxis.set_label_coords(-0.1,0.5)

for i in range(3):
    axs[i].set_frame_on(False)
    axs[i].set_xticks(np.arange(1975,2004))
    axs[i].set_xticklabels(np.arange(1975,2004),rotation='vertical')
    axs[i].set_xlim([1974.4,2003.6])
    axs[i].legend()
    axs[i].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

plt.savefig('Fig4.pdf',bbox_inches='tight')



# =============================================================================
# Figure SI.2: Groundwater pumping at different scales
# =============================================================================
fig5, axs2 = plt.subplots(3,1,figsize=(6.5,13.5))

bar_width = 0.4
axs2[0].set_title('Central Valley')
axs2[0].bar(datadroughts.wateryear, datadroughts.drought, 1.01, color = 'lightgray', label='Drought')
axs2[0].bar(datacv.wateryear[datacv.model == 'cvhm'] - bar_width/2 - 0.02, datacv.pumping[datacv.model == 'cvhm'], bar_width, label='CVHM', color = 'darkorange')
axs2[0].bar(datacv.wateryear[datacv.model == 'c2vsim'] + bar_width/2 + 0.02, datacv.pumping[datacv.model == 'c2vsim'] , bar_width, label = 'C2VSim', color = 'mediumpurple')
axs2[0].set_ylim([0,20000])
axs2[0].set_ylabel('Annual groundwater pumping\n(millions of cubic meters)')
axs2[0].yaxis.set_label_coords(-0.125,0.5)

axs2[1].set_title('Tulare Lake hydrologic region')
axs2[1].bar(datadroughts.wateryear, datadroughts.drought, 1.01, color = 'lightgray', label='Drougth')
axs2[1].bar(datatulare.wateryear[datatulare.model == 'cvhm'] - bar_width/2 - 0.02, datatulare.pumping[datatulare.model == 'cvhm'], bar_width, label='CVHM', color = 'darkorange')
axs2[1].bar(datatulare.wateryear[datatulare.model == 'c2vsim'] + bar_width/2 + 0.02, datatulare.pumping[datatulare.model == 'c2vsim'] , bar_width, label = 'C2VSim', color = 'mediumpurple')
axs2[1].set_ylim([0,12500])
axs2[1].set_ylabel('Annual groundwater pumping\n(millions of cubic meters)')
axs2[1].yaxis.set_label_coords(-0.1,0.5)

axs2[2].set_title('Sub-region 18: Kaweah and Tule Basins')
axs2[2].bar(datadroughts.wateryear, datadroughts.drought, 1.01, color = 'lightgray', label = 'Drought')
axs2[2].bar(data18.wateryear[data18.model == 'cvhm'] - bar_width/2 - 0.02, data18.pumping[data18.model == 'cvhm'], bar_width, label='CVHM', color = 'darkorange')
axs2[2].bar(data18.wateryear[data18.model == 'c2vsim'] + bar_width/2 + 0.02, data18.pumping[data18.model == 'c2vsim'] , bar_width, label = 'C2VSim', color = 'mediumpurple')
axs2[2].set_ylim([0,3000])
axs2[2].set_ylabel('Annual groundwater pumping\n(millions of cubic meters)')
axs2[2].yaxis.set_label_coords(-0.1,0.5)

for i in range(3):
    axs2[i].set_frame_on(False)
    axs2[i].set_xticks(np.arange(1975,2004))
    axs2[i].set_xticklabels(np.arange(1975,2004),rotation='vertical')
    axs2[i].set_xlim([1974.4,2003.6])
    axs2[i].legend()
    axs2[i].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

plt.savefig('FigSI2.pdf',bbox_inches='tight')




# =============================================================================
# Figure 6: Analysis of subregional variability
# =============================================================================
#Boxplot of regional variability
sns.set_style("darkgrid")
f = plt.figure(3, figsize=(8,10))

bxpl1 = f.add_subplot(4,3,1)
bxpl1 = sns.boxplot(x="surfdiv", y="region", hue="model", linewidth = 0.5, data=datasac, orient = "h", palette = "Set3", showfliers = False)
bxpl1.set(title="Surface water diversions")
bxpl1.set(xlabel='', ylabel='Sacramento\nsub-regions')
bxpl1.set(xlim=(0,3000))
bxpl1.get_legend().remove()

bxpl2 = f.add_subplot(4,3,4)
bxpl2 = sns.boxplot(x="surfdiv", y="region", hue="model", linewidth = 0.5, data=datasj, orient = "h", palette = "Set3", showfliers = False)
bxpl2.set(title="")
bxpl2.set(xlabel='', ylabel='Delta and San Joaquin River\nsub-regions')
bxpl2.set(xlim=(0,3000))
bxpl2.get_legend().remove()

bxpl3 = f.add_subplot(4,3,7)
bxpl3 = sns.boxplot(x="surfdiv", y="region", hue="model", linewidth = 0.5, data=datatul, orient = "h", palette = "Set3", showfliers = False)
bxpl3.set(title="")
bxpl3.set(xlabel='', ylabel='Tulare Lake\nsub-regions')
bxpl3.set(xlim=(0,3000))
bxpl3.get_legend().remove()

bxpl4 = f.add_subplot(4,3,10)
bxpl4 = sns.boxplot(x="surfdiv", y="basin", hue="model", linewidth = 0.5, data=databasins, orient = "h", palette = "Set3", showfliers = False, order=["Sac", "Del", "SJR", "Tul"])
bxpl4.set(title="")
bxpl4.set(xlabel='Annual surface diversions\n(mcm/year)', ylabel='Central Valley Basins')
bxpl4.set(xlim=(0,10000 ))
bxpl4.get_legend().remove()

bxpl5 = f.add_subplot(4,3,2)
bxpl5 = sns.boxplot(x="et", y="region", hue="model", linewidth = 0.5, data=datasac, orient = "h", palette = "Set3", showfliers = False)
bxpl5.set(title="Evapotranspiration")
bxpl5.set(yticklabels=[])
bxpl5.set(xlabel='', ylabel='')
bxpl5.set(xlim=(0,4000))
bxpl5.get_legend().remove()

bxpl6 = f.add_subplot(4,3,5)
bxpl6 = sns.boxplot(x="et", y="region", hue="model", linewidth = 0.5, data=datasj, orient = "h", palette = "Set3", showfliers = False)
bxpl6.set(title="")
bxpl6.set(yticklabels=[])
bxpl6.set(xlabel='', ylabel='')
bxpl6.set(xlim=(0,4000))
bxpl6.get_legend().remove()

bxpl7 = f.add_subplot(4,3,8)
bxpl7 = sns.boxplot(x="et", y="region", hue="model", linewidth = 0.5, data=datatul, orient = "h", palette = "Set3", showfliers = False)
bxpl7.set(title="")
bxpl7.set(yticklabels=[])
bxpl7.set(xlabel='', ylabel='')
bxpl7.set(xlim=(0,4000))
bxpl7.get_legend().remove()

bxpl8 = f.add_subplot(4,3,11)
bxpl8 = sns.boxplot(x="et", y="basin", hue="model", linewidth = 0.5, data=databasins, orient = "h", palette = "Set3", showfliers = False,  order=["Sac", "Del", "SJR", "Tul"])
bxpl8.set(title="")
bxpl8.set(yticklabels=[])
bxpl8.set(xlabel='Annual evapotranspiration\n(mcm/year)', ylabel='')
bxpl8.set(xlim=(0,20000))
handles1, labels1 = bxpl8.get_legend_handles_labels()
labels1 = ["CVHM", "C2VSim"]
bxpl8.legend(handles1[0:2], labels1[0:2], loc = 8, ncol=2, bbox_to_anchor=(0.5, -0.4))

bxpl9 = f.add_subplot(4,3,3)
bxpl9 = sns.boxplot(x="gwstor", y="region", hue="model", linewidth = 0.5, data=datasac, orient = "h", palette = "Set3", showfliers = False)
bxpl9.set(title="Change in groundwater storage")
bxpl9.set(yticklabels=[])
bxpl9.set(xlabel='', ylabel='')
bxpl9.set(xlim=(-2000,2000))
bxpl9.get_legend().remove()

bxpl10 = f.add_subplot(4,3,6)
bxpl10 = sns.boxplot(x="gwstor", y="region", hue="model", linewidth = 0.5, data=datasj, orient = "h", palette = "Set3", showfliers = False)
bxpl10.set(title="")
bxpl10.set(yticklabels=[])
bxpl10.set(xlabel='', ylabel='')
bxpl10.set(xlim=(-2000,2000))
bxpl10.get_legend().remove()

bxpl11 = f.add_subplot(4,3,9)
bxpl11 = sns.boxplot(x="gwstor", y="region", hue="model", linewidth = 0.5, data=datatul, orient = "h", palette = "Set3", showfliers = False)
bxpl11.set(title="")
bxpl11.set(yticklabels=[])
bxpl11.set(xlabel='', ylabel='')
bxpl11.set(xlim=(-2000,2000))
bxpl11.get_legend().remove()

bxpl12 = f.add_subplot(4,3,12)
bxpl12 = sns.boxplot(x="gwstor", y="basin", hue="model", linewidth = 0.5, data=databasins, orient = "h", palette = "Set3", showfliers = False , order=["Sac", "Del", "SJR", "Tul"])
bxpl12.set(title="")
bxpl12.set(yticklabels=[])
bxpl12.set(xlabel='Annual change in groundwater\nstorage (mcm/year)', ylabel='')
bxpl12.set(xlim=(-10000,10000))
bxpl12.get_legend().remove()

plt.savefig('Fig6.pdf',bbox_inches='tight')




# =============================================================================
# Table 1: Statistical tests
# =============================================================================
ttestresult = []
levenetest = []
vargw = []
varsw = []
shapiroc2vsim = []
shapirocvhm = []
for i in np.arange(1,22):
    drg = dataregions[dataregions.region == i]
    a = drg.gwstor[drg.model == "c2vsim"]
    b = drg.gwstor[drg.model == "cvhm"]
    ttestresult.append(ttest_rel(a,b)[1])
    levenetest.append(levene(a,b)[1])
    shapiroc2vsim.append(shapiro(a)[1])
    shapirocvhm.append(shapiro(b)[1])
    vargw.append(drg.gwstor[drg.model == "cvhm"].var())
    varsw.append(drg.surfdiv[drg.model == "cvhm"].mean())

table1 = pd.DataFrame(columns=['subregion', 'equalmeansttest', 'levenetest', 'c2vsimshapiro', 'cvhmshapiro']) #Table 1 in the paper
table1.subregion = np.arange(1,22)
table1.equalmeansttest = ttestresult
table1.levenetest = levenetest
table1.c2vsimshapiro = shapiroc2vsim
table1.cvhmshapiro = shapirocvhm

table1.to_csv('Table1.csv', index = False)

del ttestresult, levenetest, shapiroc2vsim, shapirocvhm



# =============================================================================
# Table 2: Table combining results of the two models
# =============================================================================
table2_means = dataregions.groupby(['model','region']).mean().reset_index()
table2_variances = dataregions.groupby(['model','region']).std().reset_index()
table2_combmean = dataregions.groupby(['region']).mean().reset_index()

table2 = pd.DataFrame(columns=['subregion', 'surfdiv_c2vsim_mean', 'surfdiv_c2vsim_std', 'surfdiv_cvhm_mean', 'surfdiv_cvhm_std',\
                               'surfdiv_comb_mean', 'surfdiv_comb_std','pumping_c2vsim_mean', 'pumping_c2vsim_std',\
                               'pumping_cvhm_mean', 'pumping_cvhm_std','pumping_comb_mean', 'pumping_comb_std',\
                               'gwstor_c2vsim_mean', 'gwstor_c2vsim_std','gwstor_cvhm_mean', 'gwstor_cvhm_std',\
                               'gwstor_comb_mean', 'gwstor_comb_std']) #Table 2 in the paper

table2_means = dataregions.groupby(['model','region']).mean().reset_index()
table2_std = dataregions.groupby(['model','region']).std().reset_index()
table2_combmean = dataregions.groupby(['region']).mean().reset_index()
#To obtain the variance of correlated samples we have to obtain first the covariance
covariances = []
for variable in ('surfdiv', 'pumping', 'gwstor'):
    indcov = []
    for region in np.arange(1,22):
        a = dataregions[(dataregions.region == region) & (dataregions.model == 'c2vsim')]
        b = dataregions[(dataregions.region == region) & (dataregions.model == 'cvhm')]
        indcov.append(np.cov(a[variable], b[variable])[0][1])
    covariances.append(indcov)

table2.subregion = np.arange(1,22)
table2.surfdiv_c2vsim_mean = table2_means.surfdiv[table2_means.model == 'c2vsim'].reset_index().iloc[:,1]
table2.surfdiv_c2vsim_std = table2_std.surfdiv[table2_std.model == 'c2vsim'].reset_index().iloc[:,1]
table2.surfdiv_cvhm_mean = table2_means.surfdiv[table2_means.model == 'cvhm'].reset_index().iloc[:,1]
table2.surfdiv_cvhm_std = table2_std.surfdiv[table2_std.model == 'cvhm'].reset_index().iloc[:,1]
table2.surfdiv_comb_mean = table2_combmean.surfdiv
table2['surfdiv_covariance'] = covariances[0]
table2.surfdiv_comb_std = np.sqrt((0.25 * (table2.surfdiv_c2vsim_std)* (table2.surfdiv_c2vsim_std))+(0.25 * (table2.surfdiv_cvhm_std)* (table2.surfdiv_cvhm_std))+(0.5*table2.surfdiv_covariance)) #Var((X+Y)/2)=0.25*var(X) + 0.25*var(Y) + 0.5*cov(X,Y)
table2.pumping_c2vsim_mean = table2_means.pumping[table2_means.model == 'c2vsim'].reset_index().iloc[:,1]
table2.pumping_c2vsim_std = table2_std.pumping[table2_std.model == 'c2vsim'].reset_index().iloc[:,1]
table2.pumping_cvhm_mean = table2_means.pumping[table2_means.model == 'cvhm'].reset_index().iloc[:,1]
table2.pumping_cvhm_std = table2_std.pumping[table2_std.model == 'cvhm'].reset_index().iloc[:,1]
table2.pumping_comb_mean = table2_combmean.pumping
table2['pumping_covariance'] = covariances[1]
table2.pumping_comb_std =np.sqrt((0.25 * (table2.pumping_c2vsim_std)* (table2.pumping_c2vsim_std))+(0.25 * (table2.pumping_cvhm_std)* (table2.pumping_cvhm_std))+(0.5*table2.pumping_covariance))
table2.gwstor_c2vsim_mean = table2_means.gwstor[table2_means.model == 'c2vsim'].reset_index().iloc[:,1]
table2.gwstor_c2vsim_std = table2_std.gwstor[table2_std.model == 'c2vsim'].reset_index().iloc[:,1]
table2.gwstor_cvhm_mean = table2_means.gwstor[table2_means.model == 'cvhm'].reset_index().iloc[:,1]
table2.gwstor_cvhm_std = table2_std.gwstor[table2_std.model == 'cvhm'].reset_index().iloc[:,1]
table2.gwstor_comb_mean = table2_combmean.gwstor
table2['gwstor_covariance'] = covariances[2]
table2.gwstor_comb_std = np.sqrt((0.25 * (table2.gwstor_c2vsim_std)* (table2.gwstor_c2vsim_std))+(0.25 * (table2.gwstor_cvhm_std)* (table2.gwstor_cvhm_std))+(0.5*table2.gwstor_covariance))

table2 = table2.drop(columns=['surfdiv_covariance', 'pumping_covariance','gwstor_covariance'])

table2.to_csv('Table2.csv', index = False)


# =============================================================================
# Defining auxiliar functions
# =============================================================================
#Probability of achieving sustainability
def probendoverdraft(thresholdinmeters, acreage, specyield, years, mean, variance, reduse):
    #This function calculates the probability of gw storage being above a defined threshold with respect to today's gw storage after n years
    #Threshold in meters is the change in groundwater elevation that we allow to decline from today's storage
    #Region's acreage (to calculate change in gw elevation)
    #Specific yield of unconfined aquifer (to calculate change in gw elevation)
    #Years is the number of years from today
    #Mean is the mean annual change in groundwater storage of the basin (overdraft is negative, refill positive)
    #Variance is the annual variance of groundwater storage change
    #Reduse is the reduction in groundwater use with respect to current use
    
    #If you are reading this, sorry: the paper was originally defined in the imperial system, but finally we decided to move to international. That's why there's some mixing (but it works!)
    
    threshold = thresholdinmeters * acreage * 4046.86 * specyield 
    cumvariance = years * variance
    standdev = np.sqrt(cumvariance)
    return norm.cdf(0,-(threshold*0.000001)-(years*(reduse+mean)),standdev)


# =============================================================================
# Figure 7: Probability of ending overdraft (3 subplots for different periods)
# =============================================================================
fig7 = plt.figure(4, figsize=(6.5,9))

regions = [13,14,15,16,17,18,19,20]#7,8,9,10,11,12]
ax = fig7.add_subplot(3, 1, 1) # nrows, ncols, index
ax=plt.gca()
ax.set_facecolor('w')
for i in np.arange(len(regions)):
    mu = table2.gwstor_comb_mean[regions[i]]
    sigma = table2.gwstor_comb_std[regions[i]]
    minx = mu-3.5*sigma
    maxx = mu+3.5*sigma
    regionname = "Sub-region " + str(regions[i]+1)
    acreage = dataspecyield.acreage[regions[i]]
    specyield = dataspecyield.specyield[regions[i]]
    a = []
    for j in np.arange(0, np.round(maxx)):
        xax = np.arange(0, np.round(maxx))
        elevationthres = 0
        years = 20
        variance = (table2.gwstor_comb_std[regions[i]])*(table2.gwstor_comb_std[regions[i]])
        a.append(probendoverdraft(elevationthres, acreage, specyield,years,mu,variance,j))
        
    # plot for probability distribution of overdraft
    region = regions[i]    
    plt.plot(xax, a,label=regionname)
    plt.legend(loc='lower right')
    axes = plt.gca()
    axes.set_ylim([0,1])
    axes.set_xlim([0,700])
    plt.ylabel('Probability of achieving\nsustainability after 20 years')
    plt.yticks(np.arange(0,1.05,0.1))
    
ax = fig7.add_subplot(3, 1, 2) # nrows, ncols, index
ax=plt.gca()
ax.set_facecolor('w')
for i in np.arange(len(regions)):
    mu = table2.gwstor_comb_mean[regions[i]]
    sigma = table2.gwstor_comb_std[regions[i]]
    minx = mu-3.5*sigma
    maxx = mu+3.5*sigma
    regionname = "Sub-region " + str(regions[i]+1)
    acreage = dataspecyield.acreage[regions[i]]
    specyield = dataspecyield.specyield[regions[i]]
    a = []
    for j in np.arange(0, np.round(maxx)):
        xax = np.arange(0, np.round(maxx))
        elevationthres = 0
        years = 100
        variance = (table2.gwstor_comb_std[regions[i]])*(table2.gwstor_comb_std[regions[i]])
        a.append(probendoverdraft(elevationthres, acreage, specyield,years,mu,variance,j))
        
    # plot for probability distribution of overdraft
    region = regions[i]    
    plt.plot(xax, a,label=regionname)
    plt.legend(loc='lower right')
    axes = plt.gca()
    axes.set_ylim([0,1])
    axes.set_xlim([0,700])
    plt.ylabel('Probability of achieving\nsustainability after 100 years')
    plt.yticks(np.arange(0,1.05,0.1))
    
ax = fig7.add_subplot(3, 1, 3) # nrows, ncols, index
ax=plt.gca()
ax.set_facecolor('w')
for i in np.arange(len(regions)):
    mu = table2.gwstor_comb_mean[regions[i]]
    sigma = table2.gwstor_comb_std[regions[i]]
    minx = mu-3.5*sigma
    maxx = mu+3.5*sigma
    regionname = "Sub-region " + str(regions[i]+1)
    acreage = dataspecyield.acreage[regions[i]]
    specyield = dataspecyield.specyield[regions[i]]
    a = []
    for j in np.arange(0, np.round(maxx)):
        xax = np.arange(0, np.round(maxx))
        elevationthres = 0
        years = 200
        variance = (table2.gwstor_comb_std[regions[i]])*(table2.gwstor_comb_std[regions[i]])
        a.append(probendoverdraft(elevationthres, acreage, specyield,years,mu,variance,j))
        
    # plot for probability distribution of overdraft
    region = regions[i]    
    plt.plot(xax, a,label=regionname)
    plt.legend(loc='lower right')
    axes = plt.gca()
    axes.set_ylim([0,1])
    axes.set_xlim([0,700])
    plt.xlabel('Reduction in water use (millions of cubic meters per year)')
    plt.ylabel('Probability of achieving\nsustainability after 200 years')
    plt.yticks(np.arange(0,1.05,0.1))

plt.savefig('Fig7.pdf',bbox_inches='tight')


# =============================================================================
# Figure SI3: Probability of ending overdraft (3 subplots for different periods) Sacramento
# =============================================================================
fig10 = plt.figure(5, figsize=(6.5,9))

regions = [0,1,2,3,4,5,6]#7,8,9,10,11,12]
ax = fig10.add_subplot(3, 1, 1) # nrows, ncols, index
ax=plt.gca()
ax.set_facecolor('w')
for i in np.arange(len(regions)):
    mu = table2.gwstor_comb_mean[regions[i]]
    sigma = table2.gwstor_comb_std[regions[i]]
    minx = mu-3.5*sigma
    maxx = mu+3.5*sigma
    regionname = "Sub-region " + str(regions[i]+1)
    acreage = dataspecyield.acreage[regions[i]]
    specyield = dataspecyield.specyield[regions[i]]
    a = []
    for j in np.arange(0, np.round(maxx)):
        xax = np.arange(0, np.round(maxx))
        elevationthres = 0
        years = 20
        variance = (table2.gwstor_comb_std[regions[i]])*(table2.gwstor_comb_std[regions[i]])
        a.append(probendoverdraft(elevationthres, acreage, specyield,years,mu,variance,j))
        
    # plot for probability distribution of overdraft
    region = regions[i]    
    plt.plot(xax, a,label=regionname)
    plt.legend(loc='lower right')
    axes = plt.gca()
    axes.set_ylim([0,1])
    axes.set_xlim([0,300])
    plt.ylabel('Probability of achieving\nsustainability after 20 years')
    plt.yticks(np.arange(0,1.05,0.1))
    
ax = fig10.add_subplot(3, 1, 2) # nrows, ncols, index
ax=plt.gca()
ax.set_facecolor('w')
for i in np.arange(len(regions)):
    mu = table2.gwstor_comb_mean[regions[i]]
    sigma = table2.gwstor_comb_std[regions[i]]
    minx = mu-3.5*sigma
    maxx = mu+3.5*sigma
    regionname = "Sub-region " + str(regions[i]+1)
    acreage = dataspecyield.acreage[regions[i]]
    specyield = dataspecyield.specyield[regions[i]]
    a = []
    for j in np.arange(0, np.round(maxx)):
        xax = np.arange(0, np.round(maxx))
        elevationthres = 0
        years = 100
        variance = (table2.gwstor_comb_std[regions[i]])*(table2.gwstor_comb_std[regions[i]])
        a.append(probendoverdraft(elevationthres, acreage, specyield,years,mu,variance,j))
        
    # plot for probability distribution of overdraft
    region = regions[i]    
    plt.plot(xax, a,label=regionname)
    plt.legend(loc='lower right')
    axes = plt.gca()
    axes.set_ylim([0,1])
    axes.set_xlim([0,300])
    plt.ylabel('Probability of achieving\nsustainability after 100 years')
    plt.yticks(np.arange(0,1.05,0.1))
    
ax = fig10.add_subplot(3, 1, 3) # nrows, ncols, index
ax=plt.gca()
ax.set_facecolor('w')
for i in np.arange(len(regions)):
    mu = table2.gwstor_comb_mean[regions[i]]
    sigma = table2.gwstor_comb_std[regions[i]]
    minx = mu-3.5*sigma
    maxx = mu+3.5*sigma
    regionname = "Sub-region " + str(regions[i]+1)
    acreage = dataspecyield.acreage[regions[i]]
    specyield = dataspecyield.specyield[regions[i]]
    a = []
    for j in np.arange(0, np.round(maxx)):
        xax = np.arange(0, np.round(maxx))
        elevationthres = 0
        years = 200
        variance = (table2.gwstor_comb_std[regions[i]])*(table2.gwstor_comb_std[regions[i]])
        a.append(probendoverdraft(elevationthres, acreage, specyield,years,mu,variance,j))
        
    # plot for probability distribution of overdraft
    region = regions[i]    
    plt.plot(xax, a,label=regionname)
    plt.legend(loc='lower right')
    axes = plt.gca()
    axes.set_ylim([0,1])
    axes.set_xlim([0,300])
    plt.xlabel('Reduction in water use (millions of cubic meters per year)')
    plt.ylabel('Probability of achieving\nsustainability after 200 years')
    plt.yticks(np.arange(0,1.05,0.1))

plt.savefig('FigSI3.pdf',bbox_inches='tight')



# =============================================================================
# Figure SI4: Probability of ending overdraft (3 subplots for different periods) Delta and San Joaquin
# =============================================================================
fig11 = plt.figure(6, figsize=(6.5,9))

regions = [7,8,9,10,11,12]
ax = fig11.add_subplot(3, 1, 1) # nrows, ncols, index
ax=plt.gca()
ax.set_facecolor('w')
for i in np.arange(len(regions)):
    mu = table2.gwstor_comb_mean[regions[i]]
    sigma = table2.gwstor_comb_std[regions[i]]
    minx = mu-3.5*sigma
    maxx = mu+3.5*sigma
    regionname = "Sub-region " + str(regions[i]+1)
    acreage = dataspecyield.acreage[regions[i]]
    specyield = dataspecyield.specyield[regions[i]]
    a = []
    for j in np.arange(0, np.round(maxx)):
        xax = np.arange(0, np.round(maxx))
        elevationthres = 0
        years = 20
        variance = (table2.gwstor_comb_std[regions[i]])*(table2.gwstor_comb_std[regions[i]])
        a.append(probendoverdraft(elevationthres, acreage, specyield,years,mu,variance,j))
        
    # plot for probability distribution of overdraft
    region = regions[i]    
    plt.plot(xax, a,label=regionname)
    plt.legend(loc='lower right')
    axes = plt.gca()
    axes.set_ylim([0,1])
    axes.set_xlim([0,500])
    plt.ylabel('Probability of achieving\nsustainability after 20 years')
    plt.yticks(np.arange(0,1.05,0.1))
    
ax = fig11.add_subplot(3, 1, 2) # nrows, ncols, index
ax=plt.gca()
ax.set_facecolor('w')
for i in np.arange(len(regions)):
    mu = table2.gwstor_comb_mean[regions[i]]
    sigma = table2.gwstor_comb_std[regions[i]]
    minx = mu-3.5*sigma
    maxx = mu+3.5*sigma
    regionname = "Sub-region " + str(regions[i]+1)
    acreage = dataspecyield.acreage[regions[i]]
    specyield = dataspecyield.specyield[regions[i]]
    a = []
    for j in np.arange(0, np.round(maxx)):
        xax = np.arange(0, np.round(maxx))
        elevationthres = 0
        years = 100
        variance = (table2.gwstor_comb_std[regions[i]])*(table2.gwstor_comb_std[regions[i]])
        a.append(probendoverdraft(elevationthres, acreage, specyield,years,mu,variance,j))
        
    # plot for probability distribution of overdraft
    region = regions[i]    
    plt.plot(xax, a,label=regionname)
    plt.legend(loc='lower right')
    axes = plt.gca()
    axes.set_ylim([0,1])
    axes.set_xlim([0,500])
    plt.ylabel('Probability of achieving\nsustainability after 100 years')
    plt.yticks(np.arange(0,1.05,0.1))
    
ax = fig11.add_subplot(3, 1, 3) # nrows, ncols, index
ax=plt.gca()
ax.set_facecolor('w')
for i in np.arange(len(regions)):
    mu = table2.gwstor_comb_mean[regions[i]]
    sigma = table2.gwstor_comb_std[regions[i]]
    minx = mu-3.5*sigma
    maxx = mu+3.5*sigma
    regionname = "Sub-region " + str(regions[i]+1)
    acreage = dataspecyield.acreage[regions[i]]
    specyield = dataspecyield.specyield[regions[i]]
    a = []
    for j in np.arange(0, np.round(maxx)):
        xax = np.arange(0, np.round(maxx))
        elevationthres = 0
        years = 200
        variance = (table2.gwstor_comb_std[regions[i]])*(table2.gwstor_comb_std[regions[i]])
        a.append(probendoverdraft(elevationthres, acreage, specyield,years,mu,variance,j))
        
    # plot for probability distribution of overdraft
    region = regions[i]    
    plt.plot(xax, a,label=regionname)
    plt.legend(loc='lower right')
    axes = plt.gca()
    axes.set_ylim([0,1])
    axes.set_xlim([0,500])
    plt.xlabel('Reduction in water use (millions of cubic meters per year)')
    plt.ylabel('Probability of achieving\nsustainability after 200 years')
    plt.yticks(np.arange(0,1.05,0.1))

plt.savefig('FigSI4.pdf',bbox_inches='tight')



# =============================================================================
# Figure 8: Probability of achieving sustainability with different minimum thresholds
# =============================================================================
fig8 = plt.figure(7, figsize=(6.5,5))
ax8 = fig8.add_subplot(1, 1, 1) # nrows, ncols, index
ax8=plt.gca()
ax8.set_facecolor('w')

regions = [20]
for i in np.arange(len(regions)):
    mu = table2.gwstor_comb_mean[regions[i]]
    sigma = table2.gwstor_comb_std[regions[i]]
    minx = mu-3.5*sigma
    maxx = mu+3.5*sigma
    regionname = "region" + str(regions[i]+1)
    acreage = dataspecyield.acreage[regions[i]]
    specyield = dataspecyield.specyield[regions[i]]
    a = []
    b = []
    c = []
    for j in np.arange(0, np.round(maxx)):
        xax = np.arange(0, np.round(maxx))
        years = 20
        variance = (table2.gwstor_comb_std[regions[i]])*(table2.gwstor_comb_std[regions[i]])
        a.append(probendoverdraft(0, acreage, specyield,years,mu,variance,j))
        b.append(probendoverdraft(3, acreage, specyield,years,mu,variance,j))
        c.append(probendoverdraft(6, acreage, specyield,years,mu,variance,j))
        
        
    # plot for probability distribution of overdraft
    region = regions[i]
    plt.scatter(411.13,0.5,s=10, c = 'black' ,zorder = 2)
    plt.text(416,0.5, 'WR = 411 mcm\nb = 0 m\np=0.5', va = 'top')
    plt.scatter(411.13,0.9375,s=10, c = 'black',zorder = 2)
    plt.text(416,0.9375, 'WR = 411 mcm\nb = -3 m\np=0.94', va = 'top')
    plt.scatter(163,0.5,s=10, c = 'black',zorder = 2)
    plt.text(168,0.5, 'WR = 162 mcm\nb = -6 m\np=0.5', va = 'top')
    plt.scatter(341.5,0.75,s=10, c = 'black',zorder = 2)
    plt.text(346.5,0.75, 'WR = 341 mcm\nb = -3 m\np=0.75', va = 'top')    
    plt.plot(xax, a,label= 'b = 0', zorder = 1)
    plt.plot(xax, b, label = 'b = -3 m', zorder = 1)
    plt.plot(xax, c, label = 'b = -6 m', zorder = 1)
    ax8.set_ylim([0,1])
    ax8.set_xlim([0,650])
    plt.xlabel('Reduction in water use (millions of cubic meters per year)')
    plt.ylabel('Probability of achieving\nsustainability after 20 years')
    plt.yticks(np.arange(0,1.05,0.1))

plt.legend()
    
plt.savefig('Fig8.pdf',bbox_inches='tight')



# =============================================================================
# Figure 9: Economic costs and probability of achieving sustainability
# =============================================================================
#First we define function that converts water use reductions in economic costs
def econcost(region,watreduction):
    dataregionshortage = dataeconcost[dataeconcost.region==region]
    if watreduction > 0.001*dataregionshortage.shortagemcm[len(dataregionshortage) - 1]:
        econcost = dataregionshortage.econcost[len(dataregionshortage) - 1]
        return econcost/1000000
    else:
        counter = 0
        while watreduction >= 0.001*dataregionshortage.shortagemcm[counter]:
            counter = counter +1
        if counter == 0:
            econcostmax = 0.000001*dataregionshortage.econcost[0]
            watredabove = 0.001 * dataregionshortage.shortagemcm[0]
            econcost = econcostmax * watreduction / watredabove 
            return econcost
        else:
            econcostmin = 0.000001*dataregionshortage.econcost[counter - 1]
            econcostmax = 0.000001*dataregionshortage.econcost[counter]
            watredbelow = 0.001 * dataregionshortage.shortagemcm[counter - 1]
            watredabove = 0.001 * dataregionshortage.shortagemcm[counter]
            econcost = econcostmin + ((econcostmax - econcostmin) * (watreduction-watredbelow) / (watredabove-watredbelow)) 
            return econcost    

fig9 = plt.figure(8, figsize=(6.5,9))
ax9 = fig9.add_subplot(3, 2, 1) # nrows, ncols, index
ax9=plt.gca()
ax9.set_facecolor('w')

regions = [9,14,20]

for i in np.arange(len(regions)):
    ax9 = fig9.add_subplot(3,2,(i+1)*2 - 1)
    ax9=plt.gca()
    ax9.set_facecolor('w')
    mu = table2.gwstor_comb_mean[regions[i]]
    sigma = table2.gwstor_comb_std[regions[i]]
    maxx = mu+3.5*sigma
    regionname = "region" + str(regions[i]+1)
    acreage = dataspecyield.acreage[regions[i]]
    specyield = dataspecyield.specyield[regions[i]]
    a = []
    b = []
    c = []
    d = []
    for j in np.arange(0, np.round(maxx)):
        xax = np.arange(0, np.round(maxx))
        variance = (table2.gwstor_comb_std[regions[i]])*(table2.gwstor_comb_std[regions[i]])
        a.append(probendoverdraft(0, acreage, specyield,20,mu,variance,j))
        b.append(probendoverdraft(0, acreage, specyield,100,mu,variance,j))
        c.append(probendoverdraft(0, acreage, specyield,200,mu,variance,j))
        d.append(econcost(regions[i] + 1,j))
    if i==0:
        xlim = 30
    elif i==1:
        xlim = 100
    else:
        xlim = 250
    ax9.set_xlim([0,xlim])
    ax9.set_ylim([0,1])
    plt.title('Region ' + str(regions[i] +1) + '\n'+regioname[regions[i]]+'\nb = 0 m')
    plt.plot(d,a, label = '20 yr')
    plt.plot(d,b , label = '100 yr')
    plt.plot(d,c , label = '200 yr')
    plt.xlabel('Annual economic losses (million $)')
    plt.ylabel('Probability of sustainability')
    plt.legend()
    plt.subplots_adjust(hspace=0.5)

    ax9 = fig9.add_subplot(3,2,(i+1)*2)
    ax9=plt.gca()
    ax9.set_facecolor('w')
    a = []
    b = []
    c = []
    d = []
    for j in np.arange(0, np.round(maxx)):
        xax = np.arange(0, np.round(maxx))
        variance = (table2.gwstor_comb_std[regions[i]])*(table2.gwstor_comb_std[regions[i]])
        a.append(probendoverdraft(3, acreage, specyield,20,mu,variance,j))
        b.append(probendoverdraft(3, acreage, specyield,100,mu,variance,j))
        c.append(probendoverdraft(3, acreage, specyield,200,mu,variance,j))
        d.append(econcost(regions[i] + 1,j))
    ax9.set_xlim([0,xlim])
    ax9.set_ylim([0,1])
    plt.title('Region ' + str(regions[i] +1) + '\n'+regioname[regions[i]]+'\nb = -3 m')
    plt.plot(d,a, label = '20 yr')
    plt.plot(d,b , label = '100 yr')
    plt.plot(d,c , label = '200 yr')    
    plt.xlabel('Annual economic losses (million $)')
    plt.ylabel('Probability of sustainability')
    plt.legend()
    plt.subplots_adjust(hspace=0.5)
    
plt.savefig('Fig9.pdf',bbox_inches='tight')


# =============================================================================
# SI5 Sacramento
# =============================================================================
fig12 = plt.figure(9, figsize=(10,20))
ax9 = fig12.add_subplot(7, 2, 1) # nrows, ncols, index
ax9=plt.gca()
ax9.set_facecolor('w')

regions = [0,1,2,3,4,5,6]

for i in np.arange(len(regions)):
    ax9 = fig12.add_subplot(7,2,(i+1)*2 - 1)
    ax9=plt.gca()
    ax9.set_facecolor('w')
    mu = table2.gwstor_comb_mean[regions[i]]
    sigma = table2.gwstor_comb_std[regions[i]]
    maxx = mu+3.5*sigma
    regionname = "region" + str(regions[i]+1)
    acreage = dataspecyield.acreage[regions[i]]
    specyield = dataspecyield.specyield[regions[i]]
    a = []
    b = []
    c = []
    d = []
    for j in np.arange(0, np.round(maxx)):
        xax = np.arange(0, np.round(maxx))
        variance = (table2.gwstor_comb_std[regions[i]])*(table2.gwstor_comb_std[regions[i]])
        a.append(probendoverdraft(0, acreage, specyield,20,mu,variance,j))
        b.append(probendoverdraft(0, acreage, specyield,100,mu,variance,j))
        c.append(probendoverdraft(0, acreage, specyield,200,mu,variance,j))
        d.append(econcost(regions[i] + 1,j))
    if i==0:
        xlim = 25
    elif i==1:
        xlim = 25
    else:
        xlim = 25
    ax9.set_xlim([0,xlim])
    ax9.set_ylim([0,1])
    plt.title('Region ' + str(regions[i] +1) + '\n'+regioname[regions[i]]+'\nb = 0 m')
    plt.plot(d,a, label = '20 yr')
    plt.plot(d,b , label = '100 yr')
    plt.plot(d,c , label = '200 yr')
    plt.xlabel('Annual economic losses (million $)')
    plt.ylabel('Probability of sustainability')
    plt.legend()
    plt.subplots_adjust(hspace=.8)

    ax9 = fig12.add_subplot(7,2,(i+1)*2)
    ax9=plt.gca()
    ax9.set_facecolor('w')
    a = []
    b = []
    c = []
    d = []
    for j in np.arange(0, np.round(maxx)):
        xax = np.arange(0, np.round(maxx))
        variance = (table2.gwstor_comb_std[regions[i]])*(table2.gwstor_comb_std[regions[i]])
        a.append(probendoverdraft(3, acreage, specyield,20,mu,variance,j))
        b.append(probendoverdraft(3, acreage, specyield,100,mu,variance,j))
        c.append(probendoverdraft(3, acreage, specyield,200,mu,variance,j))
        d.append(econcost(regions[i] + 1,j))
    ax9.set_xlim([0,xlim])
    ax9.set_ylim([0,1])
    plt.title('Region ' + str(regions[i] +1) + '\n'+regioname[regions[i]]+'\nb = -3 m')
    plt.plot(d,a, label = '20 yr')
    plt.plot(d,b , label = '100 yr')
    plt.plot(d,c , label = '200 yr')    
    plt.xlabel('Annual economic losses (million $)')
    plt.ylabel('Probability of sustainability')
    plt.legend()
    plt.subplots_adjust(hspace=.8)
    
plt.savefig('FigSI5.pdf',bbox_inches='tight')


# =============================================================================
# SI6 Delta and San Joaquin
# =============================================================================
fig13 = plt.figure(10, figsize=(10,16))
ax9 = fig13.add_subplot(6, 2, 1) # nrows, ncols, index
ax9=plt.gca()
ax9.set_facecolor('w')

regions = [7,8,9,10,11,12]

for i in np.arange(len(regions)):
    ax9 = fig13.add_subplot(6,2,(i+1)*2 - 1)
    ax9=plt.gca()
    ax9.set_facecolor('w')
    mu = table2.gwstor_comb_mean[regions[i]]
    sigma = table2.gwstor_comb_std[regions[i]]
    maxx = mu+3.5*sigma
    regionname = "region" + str(regions[i]+1)
    acreage = dataspecyield.acreage[regions[i]]
    specyield = dataspecyield.specyield[regions[i]]
    a = []
    b = []
    c = []
    d = []
    for j in np.arange(0, np.round(maxx)):
        xax = np.arange(0, np.round(maxx))
        variance = (table2.gwstor_comb_std[regions[i]])*(table2.gwstor_comb_std[regions[i]])
        a.append(probendoverdraft(0, acreage, specyield,20,mu,variance,j))
        b.append(probendoverdraft(0, acreage, specyield,100,mu,variance,j))
        c.append(probendoverdraft(0, acreage, specyield,200,mu,variance,j))
        d.append(econcost(regions[i] + 1,j))
    if i==0:
        xlim = 50
    elif i==1:
        xlim = 50
    else:
        xlim = 50
    ax9.set_xlim([0,xlim])
    ax9.set_ylim([0,1])
    plt.title('Region ' + str(regions[i] +1) + '\n'+regioname[regions[i]]+'\nb = 0 m')
    plt.plot(d,a, label = '20 yr')
    plt.plot(d,b , label = '100 yr')
    plt.plot(d,c , label = '200 yr')
    plt.xlabel('Annual economic losses (million $)')
    plt.ylabel('Probability of sustainability')
    plt.legend()
    plt.subplots_adjust(hspace=.8)

    ax9 = fig13.add_subplot(6,2,(i+1)*2)
    ax9=plt.gca()
    ax9.set_facecolor('w')
    a = []
    b = []
    c = []
    d = []
    for j in np.arange(0, np.round(maxx)):
        xax = np.arange(0, np.round(maxx))
        variance = (table2.gwstor_comb_std[regions[i]])*(table2.gwstor_comb_std[regions[i]])
        a.append(probendoverdraft(3, acreage, specyield,20,mu,variance,j))
        b.append(probendoverdraft(3, acreage, specyield,100,mu,variance,j))
        c.append(probendoverdraft(3, acreage, specyield,200,mu,variance,j))
        d.append(econcost(regions[i] + 1,j))
    ax9.set_xlim([0,xlim])
    ax9.set_ylim([0,1])
    plt.title('Region ' + str(regions[i] +1) + '\n'+regioname[regions[i]]+'\nb = -3 m')
    plt.plot(d,a, label = '20 yr')
    plt.plot(d,b , label = '100 yr')
    plt.plot(d,c , label = '200 yr')    
    plt.xlabel('Annual economic losses (million $)')
    plt.ylabel('Probability of sustainability')
    plt.legend()
    plt.subplots_adjust(hspace=.8)
    
plt.savefig('FigSI6.pdf',bbox_inches='tight')


# =============================================================================
# SI7 Tulare
# =============================================================================
fig14 = plt.figure(11, figsize=(10,22))
ax9 = fig14.add_subplot(8, 2, 1) # nrows, ncols, index
ax9=plt.gca()
ax9.set_facecolor('w')

regions = [13,14,15,16,17,18,19,20]

for i in np.arange(len(regions)):
    ax9 = fig14.add_subplot(8,2,(i+1)*2 - 1)
    ax9=plt.gca()
    ax9.set_facecolor('w')
    mu = table2.gwstor_comb_mean[regions[i]]
    sigma = table2.gwstor_comb_std[regions[i]]
    maxx = mu+3.5*sigma
    regionname = "region" + str(regions[i]+1)
    acreage = dataspecyield.acreage[regions[i]]
    specyield = dataspecyield.specyield[regions[i]]
    a = []
    b = []
    c = []
    d = []
    for j in np.arange(0, np.round(maxx)):
        xax = np.arange(0, np.round(maxx))
        variance = (table2.gwstor_comb_std[regions[i]])*(table2.gwstor_comb_std[regions[i]])
        a.append(probendoverdraft(0, acreage, specyield,20,mu,variance,j))
        b.append(probendoverdraft(0, acreage, specyield,100,mu,variance,j))
        c.append(probendoverdraft(0, acreage, specyield,200,mu,variance,j))
        d.append(econcost(regions[i] + 1,j))
    if i==0:
        xlim = 200
    elif i==1:
        xlim = 200
    else:
        xlim = 200
    ax9.set_xlim([0,xlim])
    ax9.set_ylim([0,1])
    plt.title('Region ' + str(regions[i] +1) + '\n'+regioname[regions[i]]+'\nb = 0 m')
    plt.plot(d,a, label = '20 yr')
    plt.plot(d,b , label = '100 yr')
    plt.plot(d,c , label = '200 yr')
    plt.xlabel('Annual economic losses (million $)')
    plt.ylabel('Probability of sustainability')
    plt.legend()
    plt.subplots_adjust(hspace=.8)

    ax9 = fig14.add_subplot(8,2,(i+1)*2)
    ax9=plt.gca()
    ax9.set_facecolor('w')
    a = []
    b = []
    c = []
    d = []
    for j in np.arange(0, np.round(maxx)):
        xax = np.arange(0, np.round(maxx))
        variance = (table2.gwstor_comb_std[regions[i]])*(table2.gwstor_comb_std[regions[i]])
        a.append(probendoverdraft(3, acreage, specyield,20,mu,variance,j))
        b.append(probendoverdraft(3, acreage, specyield,100,mu,variance,j))
        c.append(probendoverdraft(3, acreage, specyield,200,mu,variance,j))
        d.append(econcost(regions[i] + 1,j))
    ax9.set_xlim([0,xlim])
    ax9.set_ylim([0,1])
    plt.title('Region ' + str(regions[i] +1) + '\n'+regioname[regions[i]]+'\nb = -3 m')
    plt.plot(d,a, label = '20 yr')
    plt.plot(d,b , label = '100 yr')
    plt.plot(d,c , label = '200 yr')    
    plt.xlabel('Annual economic losses (million $)')
    plt.ylabel('Probability of sustainability')
    plt.legend()
    plt.subplots_adjust(hspace=.8)
    
plt.savefig('FigSI7.pdf',bbox_inches='tight')

