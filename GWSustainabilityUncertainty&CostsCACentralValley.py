# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 12:47:27 2019

@author: Alvar Escriva-Bou & Rui Hui

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

#Creating additional datasets from original data
datacv = dataregions.groupby(['model','wateryear']).sum().reset_index()
datatulare = dataregions[dataregions.region>13].groupby(['model', 'wateryear']).sum().reset_index()
data18 = dataregions[dataregions.region==18].groupby(['model', 'wateryear']).sum().reset_index()
dataprecip7503 = dataprecip[(dataprecip.wateryear>1974) & (dataprecip.wateryear<2004)]
datadroughts = datadroughts[datadroughts.wateryear<2004]
datasac = dataregions[dataregions.region<8]
datasj = dataregions[(dataregions.region>7) & (dataregions.region<14)]
datatul = dataregions[dataregions.region>13]

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

#Figure 3: Economic cost from shadow values
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

#Plotting    
selectedregions = [10,14,16,19,20,21]
fig3 = plt.figure(0, figsize=(6.5,4))
ax = fig3.add_subplot(1, 1, 1)
for region in selectedregions:
    ax.plot(0.001*dataeconcost.shortage[dataeconcost.region==region],0.000001*dataeconcost.econcost[dataeconcost.region==region])
    ax.text(0.001*dataeconcost.shortage[dataeconcost.region==region].max(), 0.000001*dataeconcost.econcost[dataeconcost.region==region].max() + 5, region)
ax.set_frame_on(False)
ax.set_xlabel('Reduction in annual net water use (taf)')
ax.set_ylabel('Economic cost (millions of $)')
plt.savefig('Fig3.pdf',bbox_inches='tight')

#Figure 4: Precipitation, surface diversions, and change in groundwater storage
fig4, axs = plt.subplots(3,1)

axs[0].bar(datadroughts.wateryear, datadroughts.drought, 1.01, color = 'lightgray', label='Drought')
axs[0].bar(dataprecip7503.wateryear, dataprecip7503.precipstatewide, label='Precipitation', color='dodgerblue')
axs[0].set_ylim([0,50])
axs[0].set_ylabel('Statewide average precipitation (inches)')
axs[0].yaxis.set_label_coords(-0.125,0.5)

bar_width = 0.4
axs[1].bar(datadroughts.wateryear, datadroughts.drought, 1.01, color = 'lightgray', label='Drougth')
axs[1].bar(datacv.wateryear[datacv.model == 'cvhm'] - bar_width/2 - 0.02, datacv.surfdiv[datacv.model == 'cvhm'], bar_width, label='CVHM', color = 'darkorange')
axs[1].bar(datacv.wateryear[datacv.model == 'c2vsim'] + bar_width/2 + 0.02, datacv.surfdiv[datacv.model == 'c2vsim'] , bar_width, label = 'C2VSim', color = 'mediumpurple')
axs[1].set_ylim([0,20000])
axs[1].set_ylabel('Total surface diversions\nin the Central Valley\n(thousands of acre-feet)')
axs[1].yaxis.set_label_coords(-0.1,0.5)

axs[2].bar(datadroughts.wateryear, datadroughts.drought, 1.01, color = 'lightgray', label = 'Drought')
axs[2].bar(datadroughts.wateryear, -datadroughts.drought, 1.01, color = 'lightgray')
axs[2].bar(datacv.wateryear[datacv.model == 'cvhm']- bar_width/2 - 0.02, datacv.gwstor[datacv.model == 'cvhm'], bar_width, label='CVHM', color = 'darkorange')
axs[2].bar(datacv.wateryear[datacv.model == 'c2vsim'] + bar_width/2 + 0.02, datacv.gwstor[datacv.model == 'c2vsim'], bar_width , label = 'C2VSim', color = 'mediumpurple')
axs[2].set_ylim([-15000,15000])
axs[2].set_ylabel('Change in annual groundwater\nstorage in the Central Valley\n(thousands of acre-feet)')
axs[2].yaxis.set_label_coords(-0.1,0.5)

for i in range(3):
    axs[i].set_frame_on(False)
    axs[i].set_xticks(np.arange(1975,2004))
    axs[i].set_xticklabels(np.arange(1975,2004),rotation='vertical')
    axs[i].set_xlim([1974.4,2003.6])
    axs[i].legend()
    axs[i].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

plt.savefig('Fig4.pdf',bbox_inches='tight')


#Figure 5: Groundwater pumping at different scales
fig5, axs2 = plt.subplots(3,1,figsize=(6.5,13.5))

bar_width = 0.4
axs2[0].set_title('Central Valley')
axs2[0].bar(datadroughts.wateryear, datadroughts.drought, 1.01, color = 'lightgray', label='Drought')
axs2[0].bar(datacv.wateryear[datacv.model == 'cvhm'] - bar_width/2 - 0.02, datacv.pumping[datacv.model == 'cvhm'], bar_width, label='CVHM', color = 'darkorange')
axs2[0].bar(datacv.wateryear[datacv.model == 'c2vsim'] + bar_width/2 + 0.02, datacv.pumping[datacv.model == 'c2vsim'] , bar_width, label = 'C2VSim', color = 'mediumpurple')
axs2[0].set_ylim([0,16000])
axs2[0].set_ylabel('Annual groundwater pumping\n(thousands of acre-feet)')
axs2[0].yaxis.set_label_coords(-0.125,0.5)

axs2[1].set_title('Tulare Lake hydrologic region')
axs2[1].bar(datadroughts.wateryear, datadroughts.drought, 1.01, color = 'lightgray', label='Drougth')
axs2[1].bar(datatulare.wateryear[datatulare.model == 'cvhm'] - bar_width/2 - 0.02, datatulare.pumping[datatulare.model == 'cvhm'], bar_width, label='CVHM', color = 'darkorange')
axs2[1].bar(datatulare.wateryear[datatulare.model == 'c2vsim'] + bar_width/2 + 0.02, datatulare.pumping[datatulare.model == 'c2vsim'] , bar_width, label = 'C2VSim', color = 'mediumpurple')
axs2[1].set_ylim([0,10000])
axs2[1].set_ylabel('Annual groundwater pumping\n(thousands of acre-feet))')
axs2[1].yaxis.set_label_coords(-0.1,0.5)

axs2[2].set_title('Sub-region 18')
axs2[2].bar(datadroughts.wateryear, datadroughts.drought, 1.01, color = 'lightgray', label = 'Drought')
axs2[2].bar(data18.wateryear[data18.model == 'cvhm'] - bar_width/2 - 0.02, data18.pumping[data18.model == 'cvhm'], bar_width, label='CVHM', color = 'darkorange')
axs2[2].bar(data18.wateryear[data18.model == 'c2vsim'] + bar_width/2 + 0.02, data18.pumping[data18.model == 'c2vsim'] , bar_width, label = 'C2VSim', color = 'mediumpurple')
axs2[2].set_ylim([0,2500])
axs2[2].set_ylabel('Annual groundwater pumping\n(thousands of acre-feet)')
axs2[2].yaxis.set_label_coords(-0.1,0.5)

for i in range(3):
    axs2[i].set_frame_on(False)
    axs2[i].set_xticks(np.arange(1975,2004))
    axs2[i].set_xticklabels(np.arange(1975,2004),rotation='vertical')
    axs2[i].set_xlim([1974.4,2003.6])
    axs2[i].legend()
    axs2[i].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

plt.savefig('Fig5.pdf',bbox_inches='tight')


#Figure 6: Analysis of subregional variability
#Boxplot of regional variability
sns.set_style("darkgrid")
f = plt.figure(3, figsize=(6.5,7.25))

bxpl1 = f.add_subplot(3,3,1)
bxpl1 = sns.boxplot(x="surfdiv", y="region", hue="model", linewidth = 0.5, data=datasac, orient = "h", palette = "Set3", showfliers = False)
bxpl1.set_title("Surface water diversions")
bxpl1.set(xlabel='', ylabel='Sacramento\nsub-regions')
bxpl1.set(xlim=(0,2500))
bxpl1.get_legend().remove()

bxpl2 = f.add_subplot(3,3,4)
bxpl2 = sns.boxplot(x="surfdiv", y="region", hue="model", linewidth = 0.5, data=datasj, orient = "h", palette = "Set3", showfliers = False)
bxpl2.set_title("")
bxpl2.set(xlabel='', ylabel='Delta and San Joaquin River\nsub-regions')
bxpl2.set(xlim=(0,2500))
bxpl2.get_legend().remove()

bxpl3 = f.add_subplot(3,3,7)
bxpl3 = sns.boxplot(x="surfdiv", y="region", hue="model", linewidth = 0.5, data=datatul, orient = "h", palette = "Set3", showfliers = False)
bxpl3.set_title("")
bxpl3.set(xlabel='Annual surface diversions\n(taf/year)', ylabel='Tulare Lake\nsub-regions')
bxpl3.set(xlim=(0,2500))
bxpl3.get_legend().remove()

bxpl4 = f.add_subplot(3,3,2)
bxpl4 = sns.boxplot(x="et", y="region", hue="model", linewidth = 0.5, data=datasac, orient = "h", palette = "Set3", showfliers = False)
bxpl4.set_title("Evapotranspiration")
bxpl4.set(xlabel='', ylabel='')
bxpl4.set(xlim=(0,3000))
bxpl4.get_legend().remove()

bxpl5 = f.add_subplot(3,3,5)
bxpl5 = sns.boxplot(x="et", y="region", hue="model", linewidth = 0.5, data=datasj, orient = "h", palette = "Set3", showfliers = False)
bxpl5.set_title("")
bxpl5.set(xlabel='', ylabel='')
bxpl5.set(xlim=(0,3000))
bxpl5.get_legend().remove()

bxpl6 = f.add_subplot(3,3,8)
bxpl6 = sns.boxplot(x="et", y="region", hue="model", linewidth = 0.5, data=datatul, orient = "h", palette = "Set3", showfliers = False)
bxpl6.set_title("")
bxpl6.set(xlabel='Annual evapotranspiration\n(taf/year)', ylabel='')
bxpl6.set(xlim=(0,3000))
handles1, labels1 = bxpl6.get_legend_handles_labels()
labels1 = ["CVHM", "C2VSim"]
bxpl6.legend(handles1[0:2], labels1[0:2], loc = 8, ncol=2, bbox_to_anchor=(0.5, -0.4))

bxpl7 = f.add_subplot(3,3,3)
bxpl7 = sns.boxplot(x="gwstor", y="region", hue="model", linewidth = 0.5, data=datasac, orient = "h", palette = "Set3", showfliers = False)
bxpl7.set_title("Change in groundwater storage")
bxpl7.set(xlabel='', ylabel='')
bxpl7.set(xlim=(-1500,1500))
bxpl7.get_legend().remove()

bxpl8 = f.add_subplot(3,3,6)
bxpl8 = sns.boxplot(x="gwstor", y="region", hue="model", linewidth = 0.5, data=datasj, orient = "h", palette = "Set3", showfliers = False)
bxpl8.set_title("")
bxpl8.set(xlabel='', ylabel='')
bxpl8.set(xlim=(-1500,1500))
bxpl8.get_legend().remove()

bxpl9 = f.add_subplot(3,3,9)
bxpl9 = sns.boxplot(x="gwstor", y="region", hue="model", linewidth = 0.5, data=datatul, orient = "h", palette = "Set3", showfliers = False)
bxpl9.set_title("")
bxpl9.set(xlabel='Annual change in groundwater\nstorage (taf/year)', ylabel='')
bxpl9.set(xlim=(-1500,1500))
bxpl9.get_legend().remove()

plt.savefig('Fig6.pdf',bbox_inches='tight')


#Table 1: Statistical tests
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

#Table 2: Table combining results of the two models
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

#Probability of achieving sustainability
def probendoverdraft(thresholdinfeet, acreage, specyield, years, mean, variance, reduse):
    #This function calculates the probability of gw storage being above a defined threshold with respect to today's gw storage after n years
    #Threshold in feet is the change in groundwater elevation that we allow to decline from today's storage (if threshold is 100 taf, we calculate the probability of having a maximum groundwater storge change of -100 taf in n years)
    #Region's acreage
    #Specific yield of unconfined aquifer
    #Years is the number of years from today
    #Mean is the mean annual change in groundwater storage of the basin (overdraft is negative, refill positive)
    #Variance is the annual variance of groundwater storage change
    #Reduse is the reduction in groundwater use with respect to current use
    threshold = thresholdinfeet * acreage * specyield 
    cumvariance = years * variance
    standdev = np.sqrt(cumvariance)
    return norm.cdf(0,-(threshold*0.001)-(years*(reduse+mean)),standdev)


#Figure 7: Probability of ending overdraft (3 subplots for different periods)
fig7 = plt.figure(4, figsize=(6.5,10))
ax = fig7.add_subplot(3, 1, 1) # nrows, ncols, index
ax=plt.gca()
ax.set_facecolor('w')

regions = [0,11,13,14,15,20]
for i in np.arange(len(regions)):
    mu = table2.gwstor_comb_mean[regions[i]]
    sigma = table2.gwstor_comb_std[regions[i]]
    minx = mu-3.5*sigma
    maxx = mu+3.5*sigma
    regionname = "region" + str(regions[i]+1)
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
    plt.text(xax[(np.abs(np.asarray(a)-0.8)).argmin()]+5,a[(np.abs(np.asarray(a)-0.8)).argmin()],'{region}'.format(region=region+1))
    axes = plt.gca()
    axes.set_ylim([0,1])
    axes.set_xlim([0,650])
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
    regionname = "region" + str(regions[i]+1)
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
    plt.text(xax[(np.abs(np.asarray(a)-0.8)).argmin()]+5,a[(np.abs(np.asarray(a)-0.8)).argmin()],'{region}'.format(region=region+1))
    axes = plt.gca()
    axes.set_ylim([0,1])
    axes.set_xlim([0,650])
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
    regionname = "region" + str(regions[i]+1)
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
    plt.text(xax[(np.abs(np.asarray(a)-0.8)).argmin()]+5,a[(np.abs(np.asarray(a)-0.8)).argmin()],'{region}'.format(region=region+1))
    axes = plt.gca()
    axes.set_ylim([0,1])
    axes.set_xlim([0,650])
    plt.xlabel('Reduction in water use (taf/year)')
    plt.ylabel('Probability of achieving\nsustainability after 200 years')
    plt.yticks(np.arange(0,1.05,0.1))
    
plt.savefig('Fig7.pdf',bbox_inches='tight')

#Figure 8: Probability of achieving sustainability with different minimum thresholds
fig8 = plt.figure(5, figsize=(6.5,5))
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
        b.append(probendoverdraft(10, acreage, specyield,years,mu,variance,j))
        c.append(probendoverdraft(20, acreage, specyield,years,mu,variance,j))
        
        
    # plot for probability distribution of overdraft
    region = regions[i]
    plt.scatter(333,0.5,s=10, c = 'black' ,zorder = 2)
    plt.text(338,0.5, 'WR = 333 taf\nb = 0 ft\np=0.5', va = 'top')
    plt.scatter(333,0.9375,s=10, c = 'black',zorder = 2)
    plt.text(338,0.9375, 'WR = 333 taf\nb = -10 ft\np=0.94', va = 'top')
    plt.scatter(130.5,0.5,s=10, c = 'black',zorder = 2)
    plt.text(135,0.5, 'WR = 130 taf\nb = -20 ft\np=0.5', va = 'top')
    plt.scatter(276,0.75,s=10, c = 'black',zorder = 2)
    plt.text(281,0.75, 'WR = 276 taf\nb = -10 ft\np=0.75', va = 'top')    
    plt.plot(xax, a,label= 'b = 0', zorder = 1)
    plt.plot(xax, b, label = 'b = -10 ft', zorder = 1)
    plt.plot(xax, c, label = 'b = -20 ft', zorder = 1)
    ax8.set_ylim([0,1])
    ax8.set_xlim([0,650])
    plt.xlabel('Reduction in water use (taf/year)')
    plt.ylabel('Probability of achieving\nsustainability after 20 years')
    plt.yticks(np.arange(0,1.05,0.1))

plt.legend()
    
plt.savefig('Fig8.pdf',bbox_inches='tight')

#Figure 9: Economic costs and probability of achieving sustainability
#First we define function that converts water use reductions in economic costs
def econcost(region,watreduction):
    dataregionshortage = dataeconcost[dataeconcost.region==region]
    if watreduction > 0.001*dataregionshortage.shortage[len(dataregionshortage) - 1]:
        econcost = dataregionshortage.econcost[len(dataregionshortage) - 1]
        return econcost/1000000
    else:
        counter = 0
        while watreduction >= 0.001*dataregionshortage.shortage[counter]:
            counter = counter +1
        if counter == 0:
            econcostmax = 0.000001*dataregionshortage.econcost[0]
            watredabove = 0.001 * dataregionshortage.shortage[0]
            econcost = econcostmax * watreduction / watredabove 
            return econcost
        else:
            econcostmin = 0.000001*dataregionshortage.econcost[counter - 1]
            econcostmax = 0.000001*dataregionshortage.econcost[counter]
            watredbelow = 0.001 * dataregionshortage.shortage[counter - 1]
            watredabove = 0.001 * dataregionshortage.shortage[counter]
            econcost = econcostmin + ((econcostmax - econcostmin) * (watreduction-watredbelow) / (watredabove-watredbelow)) 
            return econcost    

fig9 = plt.figure(6, figsize=(6.5,10))
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
    plt.text(xlim/2,0.5, 'Region ' + str(regions[i] +1) + '\nb = 0 ft', ha = 'center', va = 'center')
    plt.plot(d,a, label = '20 yr')
    plt.plot(d,b , label = '100 yr')
    plt.plot(d,c , label = '200 yr')
    plt.xlabel('Annual economic losses (million $)')
    plt.ylabel('Probability of sustainability')
    plt.legend()

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
        a.append(probendoverdraft(10, acreage, specyield,20,mu,variance,j))
        b.append(probendoverdraft(10, acreage, specyield,100,mu,variance,j))
        c.append(probendoverdraft(10, acreage, specyield,200,mu,variance,j))
        d.append(econcost(regions[i] + 1,j))
    ax9.set_xlim([0,xlim])
    ax9.set_ylim([0,1])
    plt.text(xlim/2,0.5, 'Region ' + str(regions[i] +1) + '\nb = -10 ft', ha = 'center', va = 'center')
    plt.plot(d,a, label = '20 yr')
    plt.plot(d,b , label = '100 yr')
    plt.plot(d,c , label = '200 yr')    
    plt.xlabel('Annual economic losses (million $)')
    plt.ylabel('Probability of sustainability')
    plt.legend()
    
plt.savefig('Fig9.pdf',bbox_inches='tight')