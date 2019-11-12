# Name: Alan Danque
# Date: 20190924
# Course: DSC 530
# Desc: WORK ON FINAL PROJECT
#

'''
variables:

,'GDP at market prices current US'
,'GDP per capita current US'
,'Labor force total'
,'Population growth annual pct'
,'Population total'
,'Government expenditure per tertiary student constant PPP'
,'Government expenditure per tertiary student constant US'
,'Government expenditure per tertiary student PPP'
,'Government expenditure per tertiary student US'
,'Government expenditure per tertiary student as pct of GDP per capita pct'

# Statsmodel api logistic regression: https://www.statsmodels.org/dev/examples/notebooks/generated/formulas.html
# https://www.statsmodels.org/dev/examples/notebooks/generated/ols.html

'''
from scipy.stats import poisson
from scipy import stats
from scipy.stats.stats import pearsonr
from scipy.stats import spearmanr
from collections import Counter
from scipy.stats import genpareto
from matplotlib.ticker import PercentFormatter
from matplotlib import pyplot
from statsmodels.formula.api import ols
from scipy.stats import chisquare

import statistics
import statsmodels.formula.api as smf
import pymssql
import pandas as pd
import numpy as np
import seaborn as sns
import thinkplot
import thinkstats2
import statistics
from pylab import *
showplots =0

def SpearmanCorr(xs, ys):
    xranks = pd.Series(xs).rank()
    yranks = pd.Series(ys).rank()
    return Corr(xranks, yranks)

def Corr(xs, ys):
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    meanx, varx = MeanVar(xs)
    meany, vary = MeanVar(ys)

    corr = Cov(xs, ys, meanx, meany) / np.sqrt(varx * vary)
    return corr

def Cov(xs, ys, meanx=None, meany=None):
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    if meanx is None:
        meanx = np.mean(xs)
    if meany is None:
        meany = np.mean(ys)

    cov = np.dot(xs-meanx, ys-meany) / len(xs)
    return cov

def Mode(n_num):
    n = len(n_num)
    data = Counter(n_num)
    get_mode = dict(data)
    mode = [k for k, v in get_mode.items() if v == max(list(data.values()))]
    # print("Mode of given data set is % s" % (statistics.mode(set1)))
    #### get_mode = "Mode of given data set is % s" % (statistics.mode(data))
    if len(mode) == n:
        get_mode = "No mode found"
    else:
        get_mode = "Mode is / are: " + ', '.join(map(str, mode))
    return get_mode

def Median(n_num):
    n = len(n_num)
    n_num.sort()

    if n % 2 == 0:
        median1 = n_num[n // 2]
        median2 = n_num[n // 2 - 1]
        median = (median1 + median2) / 2
    else:
        median = n_num[n // 2]
    return median

def Var(xs, mu=None, ddof=0):
    """Computes variance.

    xs: sequence of values
    mu: option known mean
    ddof: delta degrees of freedom

    returns: float
    """
    xs = np.asarray(xs)

    if mu is None:
        mu = xs.mean()

    ds = xs - mu
    return np.dot(ds, ds) / (len(xs) - ddof)

def MeanVar(xs, ddof=0):
    """Computes mean and variance.

    Based on http://stackoverflow.com/questions/19391149/
    numpy-mean-and-variance-from-single-function

    xs: sequence of values
    ddof: delta degrees of freedom

    returns: pair of float, mean and var
    """
    xs = np.asarray(xs)
    mean = xs.mean()
    s2 = Var(xs, mean, ddof)
    return mean, s2


def PrePlot(num=None, rows=None, cols=None):
    """Takes hints about what's coming.

    num: number of lines that will be plotted
    rows: number of rows of subplots
    cols: number of columns of subplots
    """
    if num:
        _Brewer.InitIter(num)

    if rows is None and cols is None:
        return

    if rows is not None and cols is None:
        cols = 1

    if cols is not None and rows is None:
        rows = 1

    # resize the image, depending on the number of rows and cols
    size_map = {(1, 1): (8, 6),
                (1, 2): (12, 6),
                (1, 3): (12, 6),
                (1, 4): (12, 5),
                (1, 5): (12, 4),
                (2, 2): (10, 10),
                (2, 3): (16, 10),
                (3, 1): (8, 10),
                (4, 1): (8, 12),
                }

    if (rows, cols) in size_map:
        fig = plt.gcf()
        fig.set_size_inches(*size_map[rows, cols])

    # create the first subplot
    if rows > 1 or cols > 1:
        ax = plt.subplot(rows, cols, 1)
        global SUBPLOT_ROWS, SUBPLOT_COLS
        SUBPLOT_ROWS = rows
        SUBPLOT_COLS = cols
    else:
        ax = plt.gca()

    return ax


def BinnedPercentiles(df):
    """Bin the data by age and plot percentiles of weight for each bin.

    df: DataFrame
    """
    bins = np.arange(10, 48, 3)
    indices = np.digitize(df.agepreg, bins)
    groups = df.groupby(indices)

    ages = [group.agepreg.mean() for i, group in groups][1:-1]
    cdfs = [thinkstats2.Cdf(group.totalwgt_lb) for i, group in groups][1:-1]

    PrePlot(3)
    for percent in [75, 50, 25]:
        print(percent)
        weights = [cdf.Percentile(percent) for cdf in cdfs]
        print(weights)
        label = '%dth' % percent
        print(label)
        thinkplot.Plot(ages, weights, label=label)

    thinkplot.Config(xlabel="Mother's age (years)",
                     ylabel='Birth weight (lbs)',
                     xlim=[14, 45], legend=True)
    thinkplot.show()


def showPMFCDF(psoutcome, psYEARVAL, psFNAME, psTITLE, psLABEL):
    # PMF PLOT
    outcomes = psoutcome
    outcomesmean = statistics.mean(outcomes)
    length = len(outcomes)
    val, cntval = np.unique(outcomes, return_counts=True)
    prop = cntval / len(outcomes)
    poisdata = np.random.poisson(outcomesmean, length)
    plt.hist(poisdata, alpha=0.5)
    plt.ylabel("Probability")
    plt.xlabel("Outcome")
    plt.title('%s Poisson PMF %s' % (psTITLE, psYEARVAL))
    FNAME = 'PMF_%s_' %psFNAME + str(psYEARVAL)
    plt.savefig("%s.png" % FNAME, dpi=100)
    if showplots == 1:
        plt.show()
    plt.close()

    # CDF PLOT
    plt.xlabel('%s' %psLABEL)
    plt.title('%s CDF %s' % (psTITLE, psYEARVAL))
    plt.hist(outcomes, normed=True, cumulative=True, label='CDF', histtype='step', alpha=0.8, color='k')
    FNAME = 'CDF_%s_' %psFNAME + str(psYEARVAL)
    plt.savefig("%s.png" % FNAME, dpi=100)
    if showplots == 1:
        plt.show()
    plt.close()


def showDistribution(YEARVAL1, YEARVAL2, YEARVAL3, YEARVAL4, psxLABEL, psyLABEL, psTITLE, psData1, psData2, psData3, psData4, psFNAME):
    STRYEARS = str(YEARVAL1) + ' ' + str(YEARVAL2) + ' ' + str(YEARVAL3) + ' ' + str(YEARVAL4)
    plt.xlabel('%s' % psxLABEL)
    plt.ylabel('%s' % psyLABEL)
    plt.title('%s %s' % (psTITLE, STRYEARS))
    plt.hist([psData1, psData2, psData3, psData4], bins=10, rwidth=0.95,
             color=['b', 'red', 'green', 'brown'],
             label=['%s' % YEARVAL1, '%s' % YEARVAL2, '%s' % YEARVAL3, '%s' % YEARVAL4])
    plt.legend()
    FNAME = 'DIST_%s_' % psFNAME + str(YEARVAL1) + '_' + str(YEARVAL2) + '_' + str(YEARVAL3) + '_' + str(YEARVAL4)
    plt.savefig("%s.png" % FNAME, dpi=100)
    if showplots == 1:
        plt.show()
    plt.close()
    print("Histogram Mean and Variance for  %s vs %s" % (psTITLE, STRYEARS))
    print(MeanVar(psData1))
    print("Tails")
    print("  Largest value:", max(psData1))
    print("  Smallest value:", min(psData1))
    print(Median(psData1))
    print(Mode(psData1))

    print(MeanVar(psData2))
    print("Histogram Mean and Variance for  %s vs %s" % (psTITLE, STRYEARS))
    print(MeanVar(psData2))
    print("Tails")
    print("  Largest value:", max(psData2))
    print("  Smallest value:", min(psData2))
    print(Median(psData2))
    print(Mode(psData2))

    print(MeanVar(psData3))
    print("Histogram Mean and Variance for  %s vs %s" % (psTITLE, STRYEARS))
    print(MeanVar(psData3))
    print("Tails")
    print("  Largest value:", max(psData3))
    print("  Smallest value:", min(psData3))
    print(Median(psData3))
    print(Mode(psData3))

    print(MeanVar(psData4))
    print("Histogram Mean and Variance for  %s vs %s" % (psTITLE, STRYEARS))
    print(MeanVar(psData4))
    print("Tails")
    print("  Largest value:", max(psData4))
    print("  Smallest value:", min(psData4))
    print(Median(psData4))
    print(Mode(psData4))


def showPMFCDFAllYrs(psoutcome, psYEARVAL, psFNAME, psTITLE, psLABEL):
    # PMF PLOT
    outcomes = psoutcome
    outcomesmean = statistics.mean(outcomes)
    length = len(outcomes)
    val, cntval = np.unique(outcomes, return_counts=True)
    prop = cntval / len(outcomes)
    poisdata = np.random.poisson(outcomesmean, length)
    plt.hist(poisdata, alpha=0.5)
    plt.ylabel("Probability")
    plt.xlabel("Outcome")
    plt.title('%s Poisson PMF %s' % (psTITLE, psYEARVAL))
    FNAME = 'PMF_%s_' %psFNAME + str(psYEARVAL)
    plt.savefig("%s.png" % FNAME, dpi=100)
    if showplots == 1:
        plt.show()
    plt.close()

    # CDF PLOT
    plt.xlabel('%s' %psLABEL)
    plt.title('%s CDF %s' % (psTITLE, psYEARVAL))
    plt.hist(outcomes, normed=True, cumulative=True, label='CDF', histtype='step', alpha=0.8, color='k')
    FNAME = 'CDF_%s_' %psFNAME + str(psYEARVAL)
    plt.savefig("%s.png" % FNAME, dpi=100)
    if showplots == 1:
        plt.show()
    plt.close()


def showDistributionAllYrs(STRYEARS, psxLABEL, psyLABEL, psTITLE, psData1, psFNAME):
    plt.xlabel('%s' % psxLABEL)
    plt.ylabel('%s' % psyLABEL)
    plt.title('%s %s' % (psTITLE, STRYEARS))
    plt.hist([psData1], bins=10, rwidth=0.95,
             color=['b'],
             label=['%s' % STRYEARS])
    plt.legend()
    FNAME = 'DIST_%s_' % psFNAME + str(STRYEARS)
    plt.savefig("%s.png" % FNAME, dpi=100)
    if showplots == 1:
        plt.show()
    plt.close()

def createDistList(dfWB, psIndicator, psYEARVAL):
    df = dfWB[(dfWB.indicator == psIndicator) & (dfWB.YEAR == psYEARVAL)].AMOUNT.astype(float)
    lsout = []
    for items1, items2 in df.iteritems():
        lsout.append(items2)
    return lsout

def createDistListAllYrs(dfWB, psIndicator, psDemocratic):
    df = dfWB[(dfWB.indicator == psIndicator) & (dfWB.Democratic == psDemocratic)].AMOUNT.astype(float)
    lsout = []
    for items1, items2 in df.iteritems():
        lsout.append(items2)
    return lsout

def find(key, dictionary):
    for k, v in dictionary.items():
        if k == key:
            yield v
        elif isinstance(v, dict):
            for result in find(key, v):
                yield result
        elif isinstance(v, list):
            for d in v:
                for result in find(key, d):
                    yield result

def exact_mc_perm_test(xs, ys, nmc):
    n, k = len(xs), 0
    print(len(xs))
    print(len(ys))
    print(mean(xs))
    print(mean(ys))
    diff = np.abs(np.mean(xs) - np.mean(ys))
    print(diff)
    zs = np.concatenate([xs, ys])
    print(zs)
    for j in range(nmc):
        np.random.shuffle(zs)
        k += diff < np.abs(np.mean(zs[:n]) - np.mean(zs[n:]))
    return k / nmc

class CorrelationPermute(thinkstats2.HypothesisTest):

    def TestStatistic(self, data):
        xs, ys = data
        test_stat = abs(thinkstats2.Corr(xs, ys))
        return test_stat

    def RunModel(self):
        xs, ys = self.data
        xs = np.random.permutation(xs)
        return xs, ys

class DiffMeansPermute(thinkstats2.HypothesisTest):

    def TestStatistic(self, data):
        group1, group2 = data
        test_stat = abs(group1.mean() - group2.mean())
        return test_stat

    def MakeModel(self):
        group1, group2 = self.data
        self.n, self.m = len(group1), len(group2)
        self.pool = np.hstack((group1, group2))

    def RunModel(self):
        np.random.shuffle(self.pool)
        data = self.pool[:self.n], self.pool[self.n:]
        return data


class PregLengthTest(thinkstats2.HypothesisTest):

    def MakeModel(self):
        firsts, others = self.data
        self.n = len(firsts)
        self.pool = np.hstack((firsts, others))

        pmf = thinkstats2.Pmf(self.pool)
        self.values = range(35, 44)
        self.expected_probs = np.array(pmf.Probs(self.values))

    def RunModel(self):
        np.random.shuffle(self.pool)
        data = self.pool[:self.n], self.pool[self.n:]
        return data

    def TestStatistic(self, data):
        firsts, others = data
        stat = self.ChiSquared(firsts) + self.ChiSquared(others)
        return stat

    def ChiSquared(self, lengths):
        hist = thinkstats2.Hist(lengths)
        observed = np.array(hist.Freqs(self.values))
        expected = self.expected_probs * len(lengths)
        stat = sum((observed - expected) ** 2 / expected)
        return stat

def RunTests(live, iters=1000):
    """Runs the tests from Chapter 9 with a subset of the data.

    live: DataFrame
    iters: how many iterations to run
    """
    n = len(live)
    firsts = live[live.birthord == 1]
    others = live[live.birthord != 1]

    # compare pregnancy lengths
    data = firsts.prglngth.values, others.prglngth.values
    ht = DiffMeansPermute(data)
    p1 = ht.PValue(iters=iters)

    data = (firsts.totalwgt_lb.dropna().values,
            others.totalwgt_lb.dropna().values)
    ht = DiffMeansPermute(data)
    p2 = ht.PValue(iters=iters)

    # test correlation
    live2 = live.dropna(subset=['agepreg', 'totalwgt_lb'])
    data = live2.agepreg.values, live2.totalwgt_lb.values
    ht = CorrelationPermute(data)
    p3 = ht.PValue(iters=iters)

    # compare pregnancy lengths (chi-squared)
    data = firsts.prglngth.values, others.prglngth.values
    ht = PregLengthTest(data)
    p4 = ht.PValue(iters=iters)

    print('%d\t%0.2f\t%0.2f\t%0.2f\t%0.2f' % (n, p1, p2, p3, p4))



def main():
    """
    Connect to my local sql server
    """
    server = "SERVERNAME_REMOVED"
    user = "USERNAME_REMOVED"
    password = "PASSWORD_REMOVED"
    conn = pymssql.connect(server, user, password, "DSC530")

    WorldBankData = []
    cursor = conn.cursor(as_dict=True)
    cursor.execute("SELECT COUNTRY, INDICATOR, YR, Democratic, AMOUNT FROM DSC530..EDATABLEOUT (nolock)")
    for row in cursor.fetchall():
            WorldBankData.append(dict([
                                ('COUNTRY', row['COUNTRY']),
                                ('indicator', row['INDICATOR']),
                                ('YEAR', row['YR']),
                                ('Democratic', row['Democratic']),
                                ('AMOUNT', row['AMOUNT']),
                                 ]))

    # Walk through each unique year and construct 7 plots of variables containing 4 years each.
    dfWB = pd.DataFrame(WorldBankData)

    WorldBankData2 = []
    cursor = conn.cursor(as_dict=True)
    cursor.execute("SELECT COUNTRY, YR, Democratic, [Population total], [Labor force total] from vwPopulationTotal_vs_LaborForceTotal where (YR between 2001 and 2011)")
    for row in cursor.fetchall():
        WorldBankData2.append(dict([
            ('COUNTRY', row['COUNTRY']),
            ('YR', row['YR']),
            ('Democratic', row['Democratic']),
            ('Population_total', row['Population total']),
            ('Labor_force_total', row['Labor force total']),
        ]))

    WorldBankData3 = []
    cursor = conn.cursor(as_dict=True)
    #cursor.execute("SELECT COUNTRY, YR, Democratic, [Government expenditure per tertiary student US], [GDP per capita current US] from vwGovernment_expenditure_per_tertiary_student_US_vs_GDP_per_capita_current_US where (YR between 2001 and 2011) and COUNTRY in (select COUNTRY from vwTop10PopulatedCountries)")
    cursor.execute("SELECT COUNTRY, YR, Democratic, [Government expenditure per tertiary student US], [GDP per capita current US] from vwGovernment_expenditure_per_tertiary_student_US_vs_GDP_per_capita_current_US where (YR between 2001 and 2011)")
    for row in cursor.fetchall():
        WorldBankData3.append(dict([
            ('COUNTRY', row['COUNTRY']),
            ('YR', row['YR']),
            ('Democratic', row['Democratic']),
            ('Government_expenditure_per_tertiary_student_US', row['Government expenditure per tertiary student US']),
            ('GDP_per_capita_current_US', row['GDP per capita current US']),
        ]))

    WorldBankData4 = []
    cursor = conn.cursor(as_dict=True)
    cursor.execute("SELECT COUNTRY, YR,Democratic, [Government expenditure per tertiary student US], [Labor force total] from vwGovernment_expenditure_per_tertiary_student_US_vs_Labor_force_total where (YR between 2001 and 2011)")
    for row in cursor.fetchall():
        WorldBankData4.append(dict([
            ('COUNTRY', row['COUNTRY']),
            ('YR', row['YR']),
            ('Democratic', row['Democratic']),
            ('Government_expenditure_per_tertiary_student_US', row['Government expenditure per tertiary student US']),
            ('Labor_force_total', row['Labor force total']),
        ]))

    WorldBankData5 = []
    cursor = conn.cursor(as_dict=True)
    cursor.execute("SELECT COUNTRY, YR, Democratic, [LaborToPopulationPct], [GDP per capita current US] from vwLabor_force_total_BY_Population_total_vs_GDP_per_capita_current_US where (YR between 2001 and 2011)")
    for row in cursor.fetchall():
        WorldBankData5.append(dict([
            ('COUNTRY', row['COUNTRY']),
            ('YR', row['YR']),
            ('Democratic', row['Democratic']),
            ('LaborToPopulationPct', row['LaborToPopulationPct']),
            ('GDP_per_capita_current_US', row['GDP per capita current US']),
        ]))

    WorldBankData6 = []
    cursor = conn.cursor(as_dict=True)
    cursor.execute("SELECT COUNTRY, YR, Democratic, LaborToPopulationPct, GDPRNK, GDP, LABORFORCE, POPULATIONTOTAL, TertiaryExpenditure from vwRegressionAnalysisDataset ")
    for row in cursor.fetchall():
        WorldBankData6.append(dict([
            ('COUNTRY', row['COUNTRY']),
            ('YR', row['YR']),
            ('Democratic', row['Democratic']),
            ('LaborToPopulationPct', row['LaborToPopulationPct']),
            ('GDPRNK', row['GDPRNK']),
            ('GDP', row['GDP']),
            ('LABORFORCE', row['LABORFORCE']),
            ('POPULATIONTOTAL', row['POPULATIONTOTAL']),
            ('TertiaryExpenditure', row['TertiaryExpenditure']),
        ]))

    conn.close()
    dfWB2 = pd.DataFrame(WorldBankData2)
    dfWB3 = pd.DataFrame(WorldBankData3)
    dfWB4 = pd.DataFrame(WorldBankData4)
    dfWB5 = pd.DataFrame(WorldBankData5)
    dfWB6 = pd.DataFrame(WorldBankData6)

    print("Regression Analysis & Prediction")
    formula = 'GDP ~ POPULATIONTOTAL + LABORFORCE + TertiaryExpenditure'
    model = smf.ols(formula, data=dfWB6)
    results = model.fit()
    print(results.summary())
    columns = ['POPULATIONTOTAL', 'LABORFORCE', 'TertiaryExpenditure']
    new = pd.DataFrame([[150000000, 70000000, 50000]], columns=columns)
    print("Predicted GDP Value")
    print(results.predict(new))


    print("Testing Correlation - Chapter 9")
    # prepare data
    data1 = dfWB6["GDP"] #20 * randn(1000) + 100
    data2 = dfWB6["TertiaryExpenditure"] #data1 + (10 * randn(1000) + 50)
    # summarize
    print('GDP data1: mean=%.3f stdv=%.3f' % (mean(data1), std(data1)))
    print('TertiaryExpenditure data2: mean=%.3f stdv=%.3f' % (mean(data2), std(data2)))
    # plot
    pyplot.scatter(data1, data2)
    pyplot.xlabel("GDP")
    pyplot.ylabel("Tertiary Expenditure")
    pyplot.title("Correlation Test GDP vs TertiaryExpenditure")
    FNAME = 'ScatterPlot_GDP_vs_TertiaryExpenditure'
    pyplot.savefig("%s.png" % FNAME, dpi=100)
    pyplot.show()

    print("Testing Covariance")
    covariance = cov(data1, data2)
    print(covariance)

    print("Testing Pearson's Correlation")
    corr, _ = pearsonr(data1, data2)
    print('Pearsons correlation: %.3f' % corr)

    print("Testing Spearmans's R Correlation")
    corr, _ = spearmanr(data1, data2)
    print('Spearmans correlation: %.3f' % corr)

    print("Mode test")
    print("GDP ", Mode(data1))
    print("Tertiary Expenditure ", Mode(data2))
    # ALL YEAR ANALYSIS

    laboroutAllYrsDemoY = createDistListAllYrs(dfWB, 'Labor force total', 'Y')
    populationoutAllYrsDemoY = createDistListAllYrs(dfWB, 'Population total', 'Y')
    gdpcapitaoutAllYrsDemoY = createDistListAllYrs(dfWB, 'GDP per capita current US', 'Y')
    gdpmarketoutAllYrsDemoY = createDistListAllYrs(dfWB, 'GDP at market prices current US', 'Y')
    tertiaryedspendoutAllYrsDemoY = createDistListAllYrs(dfWB, 'Government expenditure per tertiary student US', 'Y')
    tertiaryedspendpergdpcapitaoutAllYrsDemoY = createDistListAllYrs(dfWB,'Government expenditure per tertiary student as pct of GDP per capita pct', 'Y')

    laboroutAllYrsDemoN = createDistListAllYrs(dfWB, 'Labor force total', 'N')
    populationoutAllYrsDemoN = createDistListAllYrs(dfWB, 'Population total', 'N')
    gdpcapitaoutAllYrsDemoN = createDistListAllYrs(dfWB, 'GDP per capita current US', 'N')
    gdpmarketoutAllYrsDemoN = createDistListAllYrs(dfWB, 'GDP at market prices current US', 'N')
    tertiaryedspendoutAllYrsDemoN = createDistListAllYrs(dfWB, 'Government expenditure per tertiary student US', 'N')
    tertiaryedspendpergdpcapitaoutAllYrsDemoN = createDistListAllYrs(dfWB,'Government expenditure per tertiary student as pct of GDP per capita pct', 'N')

    p1 = laboroutAllYrsDemoY
    p2 = laboroutAllYrsDemoN
    print("Permuatation Test in Differences in Means Chapter 9")
    print(exact_mc_perm_test(p1, p2, 30000))


    showDistributionAllYrs('AllYears', 'Labor force analysis', 'Total no of countries',
                     'Labor force population pct analysis Democratic Countries', laboroutAllYrsDemoY,
                     'Labor_Force_AllYears_Democratic_')
    showPMFCDFAllYrs(laboroutAllYrsDemoY, 'AllYears', 'Labor_Force_AllYears_Democratic_', 'Labor force population pct analysis Democratic Countries',
               'Labor force analysis')

    showDistributionAllYrs('AllYears', 'Labor force analysis', 'Total no of countries',
                     'Labor force population pct analysis Non Democratic Countries', laboroutAllYrsDemoN,
                     'Labor_Force_AllYears_NonDemocratic_')
    showPMFCDFAllYrs(laboroutAllYrsDemoN, 'AllYears', 'Labor_Force_AllYears__NonDemocratic_', 'Labor force population pct analysis Non Democratic Countries',
               'Labor force analysis')


    showDistributionAllYrs('AllYears', 'Population analysis', 'Total no of countries',
                     'Population analysis Democratic Countries', populationoutAllYrsDemoY,
                     'Population_AllYears_Democratic_')
    showPMFCDFAllYrs(populationoutAllYrsDemoY, 'AllYears', 'Population_AllYears_Democratic__', 'Population analysis Democratic Countries',
               'Labor force analysis')

    showDistributionAllYrs('AllYears', 'Labor force analysis', 'Total no of countries',
                     'Population analysis Non Democratic Countries', populationoutAllYrsDemoN,
                     'Population_AllYears_NonDemocratic__')
    showPMFCDFAllYrs(populationoutAllYrsDemoN, 'AllYears', 'Population_AllYears_NonDemocratic__', 'Population analysis Non Democratic Countries',
               'Population analysis')



    showDistributionAllYrs('AllYears', 'GDP per capita current US analysis', 'Total no of countries',
                     'GDP per capita current US analysis Democratic Countries', gdpcapitaoutAllYrsDemoY,
                     'GDP per capita current US_AllYears_Democratic__')
    showPMFCDFAllYrs(gdpcapitaoutAllYrsDemoY, 'AllYears', 'GDP_per_capita_current_US_AllYears_Democratic__', 'GDP per capita current US analysis Democratic Countries',
               'Labor force analysis')

    showDistributionAllYrs('AllYears', 'Labor force analysis', 'Total no of countries',
                     'GDP per capita current US analysis Non Democratic Countries', gdpcapitaoutAllYrsDemoN,
                     'GDP per capita current US_AllYears_NonDemocratic__')
    showPMFCDFAllYrs(gdpcapitaoutAllYrsDemoN, 'AllYears', 'GDP_per_capita_current_US_AllYears_NonDemocratic__', 'GDP per capita current US analysis Non Democratic Countries',
               'GDP per capita current US analysis')



    showDistributionAllYrs('AllYears', 'GDP at market prices current US analysis', 'Total no of countries',
                     'GDP at market prices current US analysis Democratic Countries', gdpmarketoutAllYrsDemoY,
                     'GDP_at_market_prices_current_US_AllYears_Democratic__')
    showPMFCDFAllYrs(gdpmarketoutAllYrsDemoY, 'AllYears', 'GDP_at_market_prices_current_US_AllYears_Democratic__', 'GDP at market prices current US analysis Democratic Countries',
               'Labor force analysis')

    showDistributionAllYrs('AllYears', 'Labor force analysis', 'Total no of countries',
                     'GDP at market prices current US analysis Non Democratic Countries', gdpmarketoutAllYrsDemoN,
                     'GDP at market prices current US_AllYears_NonDemocratic__')
    showPMFCDFAllYrs(gdpmarketoutAllYrsDemoN, 'AllYears', 'GDP_at_market_prices_current_US_AllYears_NonDemocratic__', 'GDP at market prices current US analysis Non Democratic Countries',
               'GDP at market prices current US analysis')



    showDistributionAllYrs('AllYears', 'Government expenditure per tertiary student US analysis', 'Total no of countries',
                     'Government expenditure per tertiary student US analysis Democratic Countries', tertiaryedspendoutAllYrsDemoY,
                     'Government_expenditure_per_tertiary_student_US_AllYears_Democratic__')
    showPMFCDFAllYrs(tertiaryedspendoutAllYrsDemoY, 'AllYears', 'Government_expenditure_per_tertiary_student_US_AllYears_Democratic', 'Government expenditure per tertiary student US analysis Democratic Countries',
               'Labor force analysis')

    showDistributionAllYrs('AllYears', 'Labor force analysis', 'Total no of countries',
                     'Government expenditure per tertiary student US analysis Non Democratic Countries', tertiaryedspendoutAllYrsDemoN,
                     'Government_expenditure_per_tertiary_student_US_AllYears_NonDemocratic__')
    showPMFCDF(tertiaryedspendoutAllYrsDemoN, 'AllYears', 'Government_expenditure_per_tertiary_student_US_AllYears_NonDemocratic', 'Government expenditure per tertiary student US analysis Non Democratic Countries',
               'Government expenditure per tertiary student US analysis')


    showDistributionAllYrs('AllYears', 'Government expenditure per tertiary student as pct of GDP per capita pct analysis', 'Total no of countries',
                     'Government expenditure per tertiary student as pct of GDP per capita pct analysis Democratic Countries', tertiaryedspendpergdpcapitaoutAllYrsDemoY,
                     'Government_expenditure_per_tertiary_student_as_pct_of_GDP_per_capita_pct_AllYears_Democratic__')

    showPMFCDFAllYrs(tertiaryedspendpergdpcapitaoutAllYrsDemoY, 'AllYears', 'Government_expenditure_per_tertiary_student_as_pct_of_GDP_per_capita_pct_AllYears_Democratic__', 'Government expenditure per tertiary student as pct of GDP per capita pct analysis Democratic Countries', 'Labor force analysis')

    showDistributionAllYrs('AllYears', 'Labor force analysis', 'Total no of countries',
                     'Government expenditure per tertiary student as pct of GDP per capita pct analysis Non Democratic Countries', tertiaryedspendpergdpcapitaoutAllYrsDemoN,
                     'Government_expenditure_per_tertiary_student_as_pct_of_GDP_per_capita_pct_AllYears_NonDemocratic__')
    showPMFCDFAllYrs(tertiaryedspendpergdpcapitaoutAllYrsDemoN, 'AllYears', 'Government_expenditure_per_tertiary_student_as_pct_of_GDP_per_capita_pct_AllYears', 'Government expenditure per tertiary student as pct of GDP per capita pct analysis Non Democratic Countries',
               'Government expenditure per tertiary student as pct of GDP per capita pct analysis')


    # SHOW YEAR BY YEAR ANALYSIS
    listYEARS = dfWB.YEAR.unique()
    # Generate Distribution Histograms, PMF and CDF Plots and saving plots to file
    cnt = 1
    for i in listYEARS:
        #if i >= 1999 and i <= 2014:
        if i >= 1996 and i <= 2011:
            YEARVAL = i
            #print(i)
            #print(cnt)
            if cnt == 1:
                YEARVAL1 = YEARVAL

                # Construct Country - output by order needed
                country = dfWB[(dfWB.indicator=='Labor force total') & (dfWB.YEAR==YEARVAL1)].COUNTRY
                countryout1 = []
                for items1, items2 in country.iteritems():
                    countryout1.append(items2)

                laborout1 = createDistList(dfWB, 'Labor force total', YEARVAL1)
                populationout1 = createDistList(dfWB, 'Population total', YEARVAL1)
                gdpcapitaout1 = createDistList(dfWB, 'GDP per capita current US', YEARVAL1)
                gdpmarketout1 = createDistList(dfWB, 'GDP at market prices current US', YEARVAL1)
                tertiaryedspendout1 = createDistList(dfWB, 'Government expenditure per tertiary student US', YEARVAL1)
                tertiaryedspendpergdpcapitaout1 = createDistList(dfWB, 'Government expenditure per tertiary student as pct of GDP per capita pct', YEARVAL1)

                # Labor Plot
                a=np.array(laborout1, dtype=np.float)
                b=np.array(populationout1, dtype=np.float)
                LaborData1 = a/b

            if cnt == 2:
                #print(YEARVAL)
                YEARVAL2 = YEARVAL
                # Construct Country - output by order needed
                country = dfWB[(dfWB.indicator == 'Labor force total') & (dfWB.YEAR == YEARVAL2)].COUNTRY
                countryout2 = []
                for items1, items2 in country.iteritems():
                    countryout2.append(items2)

                laborout2 = createDistList(dfWB, 'Labor force total', YEARVAL2)
                populationout2 = createDistList(dfWB, 'Population total', YEARVAL2)
                gdpcapitaout2 = createDistList(dfWB, 'GDP per capita current US', YEARVAL2)
                gdpmarketout2 = createDistList(dfWB, 'GDP at market prices current US', YEARVAL2)
                tertiaryedspendout2 = createDistList(dfWB, 'Government expenditure per tertiary student US', YEARVAL2)
                tertiaryedspendpergdpcapitaout2 = createDistList(dfWB, 'Government expenditure per tertiary student as pct of GDP per capita pct', YEARVAL2)

                # Labor Plot
                a = np.array(laborout2, dtype=np.float)
                b = np.array(populationout2, dtype=np.float)
                LaborData2 = a / b

            if cnt == 3:
                YEARVAL3 = YEARVAL

                # Construct Country - output by order needed
                country = dfWB[(dfWB.indicator == 'Labor force total') & (dfWB.YEAR == YEARVAL3)].COUNTRY
                countryout3 = []
                for items1, items2 in country.iteritems():
                    countryout3.append(items2)

                laborout3 = createDistList(dfWB, 'Labor force total', YEARVAL3)
                populationout3 = createDistList(dfWB, 'Population total', YEARVAL3)
                gdpcapitaout3 = createDistList(dfWB, 'GDP per capita current US', YEARVAL3)
                gdpmarketout3 = createDistList(dfWB, 'GDP at market prices current US', YEARVAL3)
                tertiaryedspendout3 = createDistList(dfWB, 'Government expenditure per tertiary student US', YEARVAL3)
                tertiaryedspendpergdpcapitaout3 = createDistList(dfWB, 'Government expenditure per tertiary student as pct of GDP per capita pct', YEARVAL3)

                # Labor Plot
                a = np.array(laborout3, dtype=np.float)
                b = np.array(populationout3, dtype=np.float)
                LaborData3 = a / b

            if cnt == 4:
                YEARVAL4 = YEARVAL
                # Construct Country - output by order needed
                #country = dfWB[(dfWB.indicator == 'Labor force total') & (dfWB.YEAR == YEARVAL4)].COUNTRY
                country = dfWB[(dfWB.indicator == 'Government expenditure per tertiary student US') & (dfWB.YEAR == YEARVAL4)].COUNTRY

                countryout4 = []
                for items1, items2 in country.iteritems():
                    countryout4.append(items2)

                laborout4 = createDistList(dfWB, 'Labor force total', YEARVAL4)
                populationout4 = createDistList(dfWB, 'Population total', YEARVAL4)
                gdpcapitaout4 = createDistList(dfWB, 'GDP per capita current US', YEARVAL4)
                gdpmarketout4 = createDistList(dfWB, 'GDP at market prices current US', YEARVAL4)
                tertiaryedspendout4 = createDistList(dfWB, 'Government expenditure per tertiary student US', YEARVAL4)
                tertiaryedspendpergdpcapitaout4 = createDistList(dfWB, 'Government expenditure per tertiary student as pct of GDP per capita pct', YEARVAL4)

                # Labor Plot
                a = np.array(laborout4, dtype=np.float)
                b = np.array(populationout4, dtype=np.float)
                LaborData4 = a / b

                print("Parieto Distribution Here")
                print(countryout4)
                print(tertiaryedspendout4)
                df = pd.DataFrame({'country': tertiaryedspendout4})
                df.index = countryout4
                df = df.sort_values(by='country', ascending=False)
                df["cumpercentage"] = df["country"].cumsum() / df["country"].sum() * 100

                fig, ax = plt.subplots()
                ax.bar(df.index, df["country"], color="C0")
                ax2 = ax.twinx()
                ax2.plot(df.index, df["cumpercentage"], color="C1", marker="D", ms=7)
                ax2.yaxis.set_major_formatter(PercentFormatter())
                ax.tick_params(axis="x", rotation=90)
                ax.tick_params(axis="y", colors="C0")
                ax2.tick_params(axis="y", colors="C1")
                plt.ylabel("Tertiary Spending")
                plt.xlabel("Country")
                plt.title('%s %s' % ('Pareto Distribution Tertiary Education Spending by Country',YEARVAL4))
                plt.savefig("%s_%s.png" % ('Pareto_Distribution_Tertiary_Education_Spending_by_Country',YEARVAL4), dpi=100)
                plt.show()

                print("Second Pareto")
                country = dfWB[(dfWB.indicator == 'GDP per capita current US') & (
                            dfWB.YEAR == YEARVAL4)].COUNTRY

                countryout4 = []
                for items1, items2 in country.iteritems():
                    countryout4.append(items2)
                df = pd.DataFrame({'country': gdpcapitaout4})
                df.index = countryout4
                df = df.sort_values(by='country', ascending=False)
                df["cumpercentage"] = df["country"].cumsum() / df["country"].sum() * 100

                fig, ax = plt.subplots()
                ax.bar(df.index, df["country"], color="C0")
                ax2 = ax.twinx()
                ax2.plot(df.index, df["cumpercentage"], color="C1", marker="D", ms=7)
                ax2.yaxis.set_major_formatter(PercentFormatter())
                ax.tick_params(axis="x", rotation=90)
                ax.tick_params(axis="y", colors="C0")
                ax2.tick_params(axis="y", colors="C1")
                plt.ylabel("GDP per Capita")
                plt.xlabel("Country")
                plt.title('%s %s' % ('GDP per Capita current US by Country', YEARVAL4))
                plt.savefig("%s_%s.png" % ('GDP_per_capita_current_US_by_Country', YEARVAL4),
                            dpi=100)
                plt.show()

                # Distribution PMF & CDF plots

                showDistribution(YEARVAL1, YEARVAL2, YEARVAL3, YEARVAL4, 'Labor force analysis', 'Total no of countries',
                                 'Labor force population pct analysis', LaborData1,LaborData2,LaborData3,LaborData4,
                                 'Labor_Force_')
                showPMFCDF(LaborData1, YEARVAL1, 'Labor_Force_', 'Labor force population pct analysis',
                           'Labor force analysis')
                showPMFCDF(LaborData2, YEARVAL2, 'Labor_Force_', 'Labor force population pct analysis',
                           'Labor force analysis')
                showPMFCDF(LaborData3, YEARVAL3, 'Labor_Force_', 'Labor force population pct analysis',
                           'Labor force analysis')
                showPMFCDF(LaborData4, YEARVAL4, 'Labor_Force_', 'Labor force population pct analysis',
                           'Labor force analysis')

                print("Generating Mode Here LaborData1")
                print(Mode(LaborData1))

                showDistribution(YEARVAL1, YEARVAL2, YEARVAL3, YEARVAL4, 'Tertiary spending per student', 'Total no of countries',
                                 'Government tertiary spending analysis', tertiaryedspendout1,tertiaryedspendout2,tertiaryedspendout3,tertiaryedspendout4,
                                 'Tertiary_Spending_Per_Student_')
                showPMFCDF(tertiaryedspendout1, YEARVAL1, 'Tertiary_Spending_Per_Student_',
                           'Government tertiary spending analysis', 'Tertiary spending per student')
                showPMFCDF(tertiaryedspendout2, YEARVAL2, 'Tertiary_Spending_Per_Student_',
                           'Government tertiary spending analysis', 'Tertiary spending per student')
                showPMFCDF(tertiaryedspendout3, YEARVAL3, 'Tertiary_Spending_Per_Student_',
                           'Government tertiary spending analysis', 'Tertiary spending per student')
                showPMFCDF(tertiaryedspendout4, YEARVAL4, 'Tertiary_Spending_Per_Student_',
                           'Government tertiary spending analysis', 'Tertiary spending per student')

                showDistribution(YEARVAL1, YEARVAL2, YEARVAL3, YEARVAL4, 'GDP per capita', 'Total no of countries',
                                 'GDP per Capita Analysis', gdpcapitaout1,gdpcapitaout2,gdpcapitaout3,gdpcapitaout4,
                                 'GDP_per_capita_')
                showPMFCDF(gdpcapitaout1, YEARVAL1, 'GDP_per_capita_', 'GDP per Capita Analysis', 'GDP per capita')
                showPMFCDF(gdpcapitaout2, YEARVAL2, 'GDP_per_capita_', 'GDP per Capita Analysis', 'GDP per capita')
                showPMFCDF(gdpcapitaout3, YEARVAL3, 'GDP_per_capita_', 'GDP per Capita Analysis', 'GDP per capita')
                showPMFCDF(gdpcapitaout4, YEARVAL4, 'GDP_per_capita_', 'GDP per Capita Analysis', 'GDP per capita')

                showDistribution(YEARVAL1, YEARVAL2, YEARVAL3, YEARVAL4, 'GDP per market', 'Total no of countries',
                                 'GDP per market analysis', gdpmarketout1,gdpmarketout2,gdpmarketout3,gdpmarketout4,
                                 'GDP_per_market_')
                showPMFCDF(gdpmarketout1, YEARVAL1, 'GDP_per_market_', 'GDP per market analysis', 'GDP per market')
                showPMFCDF(gdpmarketout2, YEARVAL2, 'GDP_per_market_', 'GDP per market analysis', 'GDP per market')
                showPMFCDF(gdpmarketout3, YEARVAL3, 'GDP_per_market_', 'GDP per market analysis', 'GDP per market')
                showPMFCDF(gdpmarketout4, YEARVAL4, 'GDP_per_market_', 'GDP per market analysis', 'GDP per market')


                showDistribution(YEARVAL1, YEARVAL2, YEARVAL3, YEARVAL4, 'Population', 'Total no of countries',
                                 'Population analysis', populationout1,populationout2,populationout3,populationout4,
                                 'Population_')
                showPMFCDF(populationout1, YEARVAL1, 'Population_', 'Population analysis', 'Population')
                showPMFCDF(populationout2, YEARVAL2, 'Population_', 'Population analysis', 'Population')
                showPMFCDF(populationout3, YEARVAL3, 'Population_', 'Population analysis', 'Population')
                showPMFCDF(populationout4, YEARVAL4, 'Population_', 'Population analysis', 'Population')

                showDistribution(YEARVAL1, YEARVAL2, YEARVAL3, YEARVAL4, 'Labor', 'Total no of countries',
                                 'Labor analysis', laborout1,laborout2,laborout3,laborout4,
                                 'Labor_')
                showPMFCDF(laborout1, YEARVAL1, 'Labor_', 'Labor analysis', 'Labor')
                showPMFCDF(laborout2, YEARVAL2, 'Labor_', 'Labor analysis', 'Labor')
                showPMFCDF(laborout3, YEARVAL3, 'Labor_', 'Labor analysis', 'Labor')
                showPMFCDF(laborout4, YEARVAL4, 'Labor_', 'Labor analysis', 'Labor')

                showDistribution(YEARVAL1, YEARVAL2, YEARVAL3, YEARVAL4, 'Tertiary Spend GDP per Capita', 'Total no of countries',
                                 'Tertiary Spend GDP per Capita analysis', tertiaryedspendpergdpcapitaout1,tertiaryedspendpergdpcapitaout2,tertiaryedspendpergdpcapitaout3,tertiaryedspendpergdpcapitaout4,
                                 'Tertiary_Spend_GDP_per_capita_')
                showPMFCDF(tertiaryedspendpergdpcapitaout1, YEARVAL1, 'Tertiary_Spend_GDP_per_capita_', 'Tertiary Spend GDP per Capita analysis', 'Tertiary Spend GDP per Capita')
                showPMFCDF(tertiaryedspendpergdpcapitaout2, YEARVAL2, 'Tertiary_Spend_GDP_per_capita_',
                           'Tertiary Spend GDP per Capita analysis', 'Tertiary Spend GDP per Capita')
                showPMFCDF(tertiaryedspendpergdpcapitaout3, YEARVAL3, 'Tertiary_Spend_GDP_per_capita_',
                           'Tertiary Spend GDP per Capita analysis', 'Tertiary Spend GDP per Capita')
                showPMFCDF(tertiaryedspendpergdpcapitaout4, YEARVAL4, 'Tertiary_Spend_GDP_per_capita_',
                           'Tertiary Spend GDP per Capita analysis', 'Tertiary Spend GDP per Capita')

            if cnt != 4:
                cnt = cnt + 1
            else:
                cnt = 1

   # Generate Correlations
    dictcorrelations = {}
    rowcnt = 0
    #listYEARS = dfWB.YEAR.unique()
    for y in listYEARS:
        if y >= 1999 and y <= 2014:

            print(y)
            laborout1 = createDistList(dfWB, 'Labor force total', y)
            populationout1 = createDistList(dfWB, 'Population total', y)
            gdpcapitaout1 = createDistList(dfWB, 'GDP per capita current US', y)
            gdpmarketout1 = createDistList(dfWB, 'GDP at market prices current US', y)
            tertiaryedspendout1 = createDistList(dfWB, 'Government expenditure per tertiary student US', y)
            tertiaryedspendpergdpcapitaout1 = createDistList(dfWB, 'Government expenditure per tertiary student as pct of GDP per capita pct', y)

            a = np.array(laborout1, dtype=np.float)
            b = np.array(populationout1, dtype=np.float)
            LaborToPopulationPct = a / b

            print("populationout & laborforceout %s" %y)
            corr, p_value = spearmanr(populationout1, laborout1)
            print("Correlated test using Spearmans R")
            print(corr)
            if corr > 0.4:
                print("Strong correlation")
            if corr >= 0.2 and corr <= 0.4:
                print("Moderate correlation")
            if corr < 0.2:
                print("Weak correlation")

            scorr, pearsons_value = pearsonr(populationout1, laborout1)
            print("Correlated test using Pearsons")
            print(scorr)
            if scorr > 0.4:
                print("Strong correlation")
            if scorr >= 0.2 and corr <= 0.4:
                print("Moderate correlation")
            if scorr < 0.2:
                print("Weak correlation")

            rowcnt = rowcnt + 1
            dictcorrelations[rowcnt]={}
            dictcorrelations[rowcnt]['Year'] = y
            dictcorrelations[rowcnt]['Indicator1'] = 'Population total'
            dictcorrelations[rowcnt]['Indicator2'] = 'Labor force total'
            dictcorrelations[rowcnt]['PCor'] = corr
            dictcorrelations[rowcnt]['SCor'] = scorr

            print(" ")
            print("LaborToPopulationPct & gdpcapitaout1 %s" %y)
            corr, p_value = spearmanr(LaborToPopulationPct, gdpcapitaout1)
            print("Correlated test using Spearmans R")
            print(corr)
            if corr > 0.4:
                print("Strong correlation")
            if corr >= 0.2 and corr <= 0.4:
                print("Moderate correlation")
            if corr < 0.2:
                print("Weak correlation")

            scorr, pearsons_value = pearsonr(LaborToPopulationPct, gdpcapitaout1)
            print("Correlated test using Pearsons")
            print(scorr)
            if scorr > 0.4:
                print("Strong correlation")
            if scorr >= 0.2 and corr <= 0.4:
                print("Moderate correlation")
            if scorr < 0.2:
                print("Weak correlation")

            rowcnt = rowcnt + 1
            dictcorrelations[rowcnt]={}
            dictcorrelations[rowcnt]['Year'] = y
            dictcorrelations[rowcnt]['Indicator1'] = 'LaborToPopulationPct Calc'
            dictcorrelations[rowcnt]['Indicator2'] = 'GDP per capita current US'
            dictcorrelations[rowcnt]['PCor'] = corr
            dictcorrelations[rowcnt]['SCor'] = scorr

            print(" ")
            print("tertiaryedspendout1 & gdpcapitaout1 %s" %y)
            corr, p_value = spearmanr(tertiaryedspendout1, gdpcapitaout1)
            print("Correlated test using Spearmans R")
            print(corr)
            if corr > 0.4:
                print("Strong correlation")
            if corr >= 0.2 and corr <= 0.4:
                print("Moderate correlation")
            if corr < 0.2:
                print("Weak correlation")

            scorr, pearsons_value = pearsonr(tertiaryedspendout1, gdpcapitaout1)
            print("Correlated test using Pearsons")
            print(corr)
            if scorr > 0.4:
                print("Strong correlation")
            if scorr >= 0.2 and corr <= 0.4:
                print("Moderate correlation")
            if scorr < 0.2:
                print("Weak correlation")
            rowcnt = rowcnt + 1
            dictcorrelations[rowcnt]={}
            dictcorrelations[rowcnt]['Year'] = y
            dictcorrelations[rowcnt]['Indicator1'] = 'Government expenditure per tertiary student US'
            dictcorrelations[rowcnt]['Indicator2'] = 'GDP per capita current US'
            dictcorrelations[rowcnt]['PCor'] = corr
            dictcorrelations[rowcnt]['SCor'] = scorr


            print(" ")
            print("tertiaryedspendout1 & gdpmarketout1 %s" %y)
            corr, p_value = spearmanr(tertiaryedspendout1, gdpmarketout1)
            print("Correlated test using Spearmans R")
            print(corr)
            if corr > 0.4:
                print("Strong correlation")
            if corr >= 0.2 and corr <= 0.4:
                print("Moderate correlation")
            if corr < 0.2:
                print("Weak correlation")

            scorr, pearsons_value = pearsonr(tertiaryedspendout1, gdpmarketout1)
            print("Correlated test using Pearsons")
            print(scorr)
            if scorr > 0.4:
                print("Strong correlation")
            if scorr >= 0.2 and corr <= 0.4:
                print("Moderate correlation")
            if scorr < 0.2:
                print("Weak correlation")
            rowcnt = rowcnt + 1
            dictcorrelations[rowcnt]={}
            dictcorrelations[rowcnt]['Year'] = y
            dictcorrelations[rowcnt]['Indicator1'] = 'Government expenditure per tertiary student US'
            dictcorrelations[rowcnt]['Indicator2'] = 'GDP at market prices current US'
            dictcorrelations[rowcnt]['PCor'] = corr
            dictcorrelations[rowcnt]['SCor'] = scorr


            print(" ")
            print("tertiaryedspendout1 & laborout1 %s" %y)
            corr, p_value = spearmanr(tertiaryedspendout1, laborout1)
            print("Correlated test using Spearmans R")
            print(corr)
            if corr > 0.4:
                print("Strong correlation")
            if corr >= 0.2 and corr <= 0.4:
                print("Moderate correlation")
            if corr < 0.2:
                print("Weak correlation")

            scorr, pearsons_value = pearsonr(tertiaryedspendout1, laborout1)
            print("Correlated test using Pearsons")
            print(scorr)
            if scorr > 0.4:
                print("Strong correlation")
            if scorr >= 0.2 and corr <= 0.4:
                print("Moderate correlation")
            if scorr < 0.2:
                print("Weak correlation")
            rowcnt = rowcnt + 1
            dictcorrelations[rowcnt]={}
            dictcorrelations[rowcnt]['Year'] = y
            dictcorrelations[rowcnt]['Indicator1'] = 'Government expenditure per tertiary student US'
            dictcorrelations[rowcnt]['Indicator2'] = 'Labor force total'
            dictcorrelations[rowcnt]['PCor'] = corr
            dictcorrelations[rowcnt]['SCor'] = scorr

            print(" ")
            print("populationout1 & tertiaryedspendpergdpcapitaout1 %s" %y)
            corr, p_value = spearmanr(populationout1, tertiaryedspendpergdpcapitaout1)
            print("Correlated test using Spearmans R")
            print(corr)
            if corr > 0.4:
                print("Strong correlation")
            if corr >= 0.2 and corr <= 0.4:
                print("Moderate correlation")
            if corr < 0.2:
                print("Weak correlation")

            scorr, pearsons_value = pearsonr(populationout1, tertiaryedspendpergdpcapitaout1)
            print("Correlated test using Pearsons")
            print(scorr)
            if scorr > 0.4:
                print("Strong correlation")
            if scorr >= 0.2 and corr <= 0.4:
                print("Moderate correlation")
            if scorr < 0.2:
                print("Weak correlation")
            rowcnt = rowcnt + 1
            dictcorrelations[rowcnt]={}
            dictcorrelations[rowcnt]['Year'] = y
            dictcorrelations[rowcnt]['Indicator1'] = 'Population total'
            dictcorrelations[rowcnt]['Indicator2'] = 'Government expenditure per tertiary student as pct of GDP per capita pct'
            dictcorrelations[rowcnt]['PCor'] = corr
            dictcorrelations[rowcnt]['SCor'] = scorr

    print("Listing those variables with more than 0.20 / -0.20 correlations")
    for i, i_info in dictcorrelations.items():
        for k in i_info:
            if i_info['SCor'] >= 0.20 or i_info['SCor'] <= -0.20:
                print(i_info[k])

    print("PLOTS START HERE")
    y=2014
    df1 = dfWB[(dfWB.indicator == 'Population total') & (dfWB.YEAR == y)].AMOUNT.astype(float)
    df2 = dfWB[(dfWB.indicator == 'Labor force total') & (dfWB.YEAR == y)].AMOUNT.astype(float)
    df3 = dfWB[(dfWB.indicator == 'GDP at market prices current US') & (dfWB.YEAR == y)].AMOUNT.astype(float)
    df4 = dfWB[(dfWB.indicator == 'GDP per capita current US') & (dfWB.YEAR == y)].AMOUNT.astype(float)
    df5 = dfWB[(dfWB.indicator == 'Government expenditure per tertiary student as pct of GDP per capita pct') & (dfWB.YEAR == y)].AMOUNT.astype(float)

    df6 = dfWB[(dfWB.indicator == 'Government expenditure per tertiary student constant PPP') & (dfWB.YEAR == y)].AMOUNT.astype(float)
    df7 = dfWB[(dfWB.indicator == 'Government expenditure per tertiary student constant US') & (dfWB.YEAR == y)].AMOUNT.astype(float)
    df8 = dfWB[(dfWB.indicator == 'Government expenditure per tertiary student PPP') & (dfWB.YEAR == y)].AMOUNT.astype(float)
    df9 = dfWB[(dfWB.indicator == 'Government expenditure per tertiary student US') & (dfWB.YEAR == y)].AMOUNT.astype(float)
    df10 = dfWB[(dfWB.indicator == 'Population growth annual pct') & (dfWB.YEAR == y)].AMOUNT.astype(float)
    df11 = dfWB5[(dfWB5.YR == y)].LaborToPopulationPct.astype(float)
    df12 = dfWB5[(dfWB5.YR == y)].GDP_per_capita_current_US.astype(float)

    dfcountry = dfWB[(dfWB.indicator == 'Labor force total') & (dfWB.YEAR == y)].COUNTRY

    # plot1 Done
    sns.lmplot(x="Population_total", y="Labor_force_total", col="Democratic", row="YR", data=dfWB2)
    #sns.plt.title('Top 10 Populated Countries & Related Labor Force Analysis %s' % y)
    sns.plt.xlabel('Population')
    sns.plt.ylabel('Labor Force')
    FNAME = 'Scatter_Top_10_%s_' % 'PopulationTotal_vs_LaborForceTotal_' + str(y)
    sns.plt.savefig("%s.png" % FNAME, dpi=100)
    sns.plt.show()

    sns.regplot(x=df1, y=df2, line_kws={"color": "r", "alpha": 0.7, "lw": 5})
    sns.plt.title('Population & Labor Force Analysis %s' % y)
    sns.plt.xlabel('Population')
    sns.plt.ylabel('Labor Force')
    FNAME = 'Scatter_%s_' % 'PopulationTotal_vs_LaborForceTotal_' + str(y)
    sns.plt.savefig("%s.png" % FNAME, dpi=100)
    sns.plt.show()
    print("PLOT1 Done")

    # plot2
    sns.lmplot(x="GDP_per_capita_current_US", y="Government_expenditure_per_tertiary_student_US", col="Democratic", row="YR", data=dfWB3)
    sns.plt.xlabel('Government expenditure per tertiary student US')
    sns.plt.ylabel('GDP per capita current US')
    FNAME = 'Scatter_%s_' % 'Government_expenditure_per_tertiary_student_US_vs_GDP_per_capita_current_US_' + str(y)
    sns.plt.savefig("%s.png" % FNAME, dpi=100)
    sns.plt.show()

    sns.regplot(x=df4, y=df9, line_kws={"color": "r", "alpha": 0.7, "lw": 5})
    sns.plt.title('Government expenditure per tertiary student US & GDP per capita current US %s' % y)
    sns.plt.xlabel('Government expenditure per tertiary student US')
    sns.plt.ylabel('GDP per capita current US')
    FNAME = 'Scatter_%s_' % 'Government_expenditure_per_tertiary_student_US_vs_GDP_per_capita_current_US_' + str(y)
    sns.plt.savefig("%s.png" % FNAME, dpi=100)
    sns.plt.show()
    print("PLOT2 Done")

    # plot3
    sns.lmplot(x="Labor_force_total", y="Government_expenditure_per_tertiary_student_US", col="Democratic", row="YR", data=dfWB4)
    sns.plt.xlabel('Government expenditure per tertiary student US')
    sns.plt.ylabel('Labor Force')
    FNAME = 'Scatter__%s_' % 'Government_expenditure_per_tertiary_student_US_vs_LaborForceTotal_' + str(y)
    sns.plt.savefig("%s.png" % FNAME, dpi=100)
    sns.plt.show()

    sns.regplot(x=df9, y=df2, line_kws={"color": "r", "alpha": 0.7, "lw": 5})
    sns.plt.title('Government expenditure per tertiary student US vs Labor %s' % y)
    sns.plt.xlabel('Government expenditure per tertiary student US')
    sns.plt.ylabel('Labor Force')
    FNAME = 'Scatter_%s_' % '_Government_expenditure_per_tertiary_student_US_vs_LaborForceTotal_' + str(y)
    sns.plt.savefig("%s.png" % FNAME, dpi=100)
    sns.plt.show()
    print("PLOT3 Done")

    # plot4 Done
    sns.lmplot(x="GDP_per_capita_current_US", y="LaborToPopulationPct", col="Democratic", row="YR", data=dfWB5)
    sns.plt.xlabel('LaborToPopulationPct')
    sns.plt.ylabel('GDP per capita current US')
    FNAME = 'Scatter_%s_' % 'LaborToPopulationPct_vs_GDP_per_capita_current_US_' + str(y)
    sns.plt.savefig("%s.png" % FNAME, dpi=100)
    sns.plt.show()

    sns.regplot(x=df12, y=df11, line_kws={"color": "r", "alpha": 0.7, "lw": 5})
    sns.plt.title('Population & Labor Force Analysis %s' % y)
    sns.plt.xlabel('LaborToPopulationPct')
    sns.plt.ylabel('GDP per capita current US')
    FNAME = 'Scatter_%s_' % '__LaborToPopulationPct_vs_GDP_per_capita_current_US_' + str(y)
    sns.plt.savefig("%s.png" % FNAME, dpi=100)
    sns.plt.show()
    print("PLOT4 Done")

    print("Covariance Tests")
    x = df12
    y = df11
    cov_mat = np.stack((x, y), axis=1)
    print("shape of matrix x and y:", np.shape(cov_mat))
    print("shape of covariance matrix:", np.shape(np.cov(cov_mat)))
    print(np.cov(cov_mat))

    print("Hypothesis Test Two-Sided Test")
    rvs1 = df12
    rvs2 = df11
    print(stats.ttest_ind(rvs1, rvs2))
    print(stats.ttest_ind(rvs1, rvs2, equal_var=False))



if __name__ == '__main__':
    main()
