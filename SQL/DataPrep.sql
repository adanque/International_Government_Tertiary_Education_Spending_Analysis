
drop table EDATABLETOREVIEW
select EDAREVIEW.* 
	into EDATABLETOREVIEW
	from EDAREVIEW where country in (select country from #REVIEW0)
		and indicator in 
(
'GDP at market prices current US'
,'GDP per capita current US'
,'Labor force total'
,'Population growth annual pct'
,'Population total'
,'Government expenditure per tertiary student constant PPP'
,'Government expenditure per tertiary student constant US'
,'Government expenditure per tertiary student PPP'
,'Government expenditure per tertiary student US'
,'Government expenditure per tertiary student as pct of GDP per capita pct'
)



--out of the list there 46 countries with 9 of the 11 indicators feasible

select country, count(*) cnt
	into #REVIEW0
from EDAREVIEW 
	where indicator in
(
 'Government expenditure per tertiary student US'
,'Government expenditure per tertiary student constant US'
,'Government expenditure per tertiary student PPP'
,'Government expenditure per secondary student US'
,'Government expenditure per tertiary student constant PPP'
,'Government expenditure per secondary student constant US'
,'Government expenditure per secondary student PPP'
,'Government expenditure per secondary student constant PPP'
,'GDP at market prices current US'
,'GDP per capita current US'
,'Population growth annual pct'
,'Population total'
,'Labor force total'
,'Government expenditure per tertiary student as pct of GDP per capita pct' --++
)
	group by country
		having count(*) >= 9




set quoted_identifier off
select distinct "select '"+ indicator +"' as indicator, count(*) cnt from EDAREVIEW where indicator = '"+ indicator +"'  union " from EDAREVIEW where cnts >= 4 group by indicator 

select * from INDICATORCNTS order by cnt desc

sp_rename INDICATORCNTS, INDICATORCNTSOLD
create table INDICATORCNTS (indicator varchar(255), cnt int)
insert into INDICATORCNTS (indicator, cnt)
select 'GDP at market prices current US' as indicator, count(*) cnt from EDAREVIEW where indicator = 'GDP at market prices current US'  union 
select 'GDP per capita current US' as indicator, count(*) cnt from EDAREVIEW where indicator = 'GDP per capita current US'  union 
select 'Government expenditure per primary student constant PPP' as indicator, count(*) cnt from EDAREVIEW where indicator = 'Government expenditure per primary student constant PPP'  union 
select 'Government expenditure per primary student constant US' as indicator, count(*) cnt from EDAREVIEW where indicator = 'Government expenditure per primary student constant US'  union 
select 'Government expenditure per primary student PPP' as indicator, count(*) cnt from EDAREVIEW where indicator = 'Government expenditure per primary student PPP'  union 
select 'Government expenditure per primary student US' as indicator, count(*) cnt from EDAREVIEW where indicator = 'Government expenditure per primary student US'  union 
select 'Government expenditure per secondary student constant PPP' as indicator, count(*) cnt from EDAREVIEW where indicator = 'Government expenditure per secondary student constant PPP'  union 
select 'Government expenditure per secondary student constant US' as indicator, count(*) cnt from EDAREVIEW where indicator = 'Government expenditure per secondary student constant US'  union 
select 'Government expenditure per secondary student PPP' as indicator, count(*) cnt from EDAREVIEW where indicator = 'Government expenditure per secondary student PPP'  union 
select 'Government expenditure per secondary student US' as indicator, count(*) cnt from EDAREVIEW where indicator = 'Government expenditure per secondary student US'  union 
select 'Government expenditure per tertiary student constant PPP' as indicator, count(*) cnt from EDAREVIEW where indicator = 'Government expenditure per tertiary student constant PPP'  union 
select 'Government expenditure per tertiary student constant US' as indicator, count(*) cnt from EDAREVIEW where indicator = 'Government expenditure per tertiary student constant US'  union 
select 'Government expenditure per tertiary student PPP' as indicator, count(*) cnt from EDAREVIEW where indicator = 'Government expenditure per tertiary student PPP'  union 
select 'Government expenditure per tertiary student US' as indicator, count(*) cnt from EDAREVIEW where indicator = 'Government expenditure per tertiary student US'  union 
select 'Illiterate population 25 to 64 years both sexes number' as indicator, count(*) cnt from EDAREVIEW where indicator = 'Illiterate population 25 to 64 years both sexes number'  union 
select 'Labor force total' as indicator, count(*) cnt from EDAREVIEW where indicator = 'Labor force total'  union 
select 'Literacy rate population 25 to 64 years both sexes pct' as indicator, count(*) cnt from EDAREVIEW where indicator = 'Literacy rate population 25 to 64 years both sexes pct'  union 
select 'Population growth annual pct' as indicator, count(*) cnt from EDAREVIEW where indicator = 'Population growth annual pct'  union 
select 'Population total' as indicator, count(*) cnt from EDAREVIEW where indicator = 'Population total'  union 
select 'UIS Percentage of population age 25 with no schooling Female' as indicator, count(*) cnt from EDAREVIEW where indicator = 'UIS Percentage of population age 25 with no schooling Female'  union 
select 'UIS Percentage of population age 25 with no schooling Male' as indicator, count(*) cnt from EDAREVIEW where indicator = 'UIS Percentage of population age 25 with no schooling Male'  union 
select 'UIS Percentage of population age 25 with no schooling Total' as indicator, count(*) cnt from EDAREVIEW where indicator = 'UIS Percentage of population age 25 with no schooling Total' union
select 'Government expenditure per tertiary student as pct of GDP per capita pct' as indicator, count(*) cnt from EDAREVIEW where indicator = 'Government expenditure per tertiary student as pct of GDP per capita pct' 
select * from INDICATORCNTS




--drop table EDAREVIEW 
select 
identity(int, 1, 1) as rowid,
indicator, [country name] COUNTRY, COUNTRYINDICATORCNTS.cnts,
[1970] YR_1970,
[1971] YR_1971,
[1972] YR_1972,
[1973] YR_1973,
[1974] YR_1974,
[1975] YR_1975,
[1976] YR_1976,
[1977] YR_1977,
[1978] YR_1978,
[1979] YR_1979,
[1980] YR_1980,
[1981] YR_1981,
[1982] YR_1982,
[1983] YR_1983,
[1984] YR_1984,
[1985] YR_1985,
[1986] YR_1986,
[1987] YR_1987,
[1988] YR_1988,
[1989] YR_1989,
[1990] YR_1990,
[1991] YR_1991,
[1992] YR_1992,
[1993] YR_1993,
[1994] YR_1994,
[1995] YR_1995,
[1996] YR_1996,
[1997] YR_1997,
[1998] YR_1998,
[1999] YR_1999,
[2000] YR_2000,
[2001] YR_2001,
[2002] YR_2002,
[2003] YR_2003,
[2004] YR_2004,
[2005] YR_2005,
[2006] YR_2006,
[2007] YR_2007,
[2008] YR_2008,
[2009] YR_2009,
[2010] YR_2010,
[2011] YR_2011,
[2012] YR_2012,
[2013] YR_2013,
[2014] YR_2014,
[2015] YR_2015,
[2016] YR_2016
--[2017] YR_2017
	into EDAREVIEW
		from EDATABLE 
			join COUNTRYINDICATORCNTS 
				on COUNTRYINDICATORCNTS.countryname = EDATABLE.[country name]
			where COUNTRYINDICATORCNTS.cnts >= 4
		---group by indicator, [country name], COUNTRYINDICATORCNTS.cnts 
			order by COUNTRYINDICATORCNTS.cnts, [country name], indicator





create table FTlist (indicator varchar(255), seqid int)
insert into FTlist (indicator , seqid )
select 
'GDP at market prices current US',1
union
select
'GDP per capita current US',2
union
select
'Government expenditure per primary student constant PPP',3
union
select
'Government expenditure per primary student constant US',4
union
select
'Government expenditure per primary student PPP',5
union
select
'Government expenditure per primary student US',6
union
select
'Government expenditure per secondary student constant PPP',7
union
select
'Government expenditure per secondary student constant US',8
union
select
'Government expenditure per secondary student PPP',9
union
select
'Government expenditure per secondary student US',10
union
select
'Government expenditure per tertiary student constant PPP',11
union
select
'Government expenditure per tertiary student constant US',12
union
select
'Government expenditure per tertiary student PPP',13
union
select
'Government expenditure per tertiary student US',14
union
select
'Labor force total',15
union
select
'Population growth annual pct',16
union
select
'Illiterate population 25 to 64 years both sexes number',17
union
select
'Literacy rate population 25 to 64 years both sexes pct',18
union
select
'Population total',19
union
select
'UIS Percentage of population age 25 with no schooling Female',20
union
select
'UIS Percentage of population age 25 with no schooling Male',21
union
select
'UIS Percentage of population age 25 with no schooling Total',22
union
select 'Government expenditure per tertiary student as pct of GDP per capita pct',23




if object_id('tempdb..#YRS') is not null drop table #YRS
select identity(int, 1, 1) as rows, 1 jnr, name yr_field, replace(name, 'YR_', '') YRNAME
	into #YRS	
from syscolumns where id = object_id('EDATABLETOREVIEW') and name like '%YR%'
select * from #YRS


if object_id('tempdb..#INDICATORS') is not null drop table #INDICATORS
select distinct identity(int, 1, 1) as rowid, 1 jnr, indicator 
	into #INDICATORS
from DSC530..EDATABLETOREVIEW 
select * from #INDICATORS


if object_id('tempdb..#COUNTRIES') is not null drop table #COUNTRIES
select distinct identity(int, 1, 1) as rowid, 1 jnr, COUNTRY
	into #COUNTRIES
from DSC530..EDATABLETOREVIEW 
select * from #COUNTRIES




if object_id('tempdb..#QUERIES') is not null drop table #QUERIES
select identity(int, 1, 1) as rowid, a.indicator, b.yr_field, c.COUNTRY 
	into #QUERIES
	from #INDICATORS a
		join #YRS b on a.jnr = b.jnr
		join #COUNTRIES c on a.jnr = c.jnr
	order by c.COUNTRY, b.yr_field, a.indicator
	select * from #QUERIES



set quoted_identifier off 
if object_id('tempdb..#GENQUERIES') is not null drop table #GENQUERIES
select 
	 identity(int, 1, 1) as rowid
	,"insert into EDATABLEOUT (COUNTRY, INDICATOR, YR, AMOUNT) select '"+ a.COUNTRY + "' as COUNTRY, '" + a.indicator + "' as INDICATOR, '" + replace(a.yr_field, 'YR_', '') + "' as YR, "+a.yr_field+" as AMOUNT from EDATABLETOREVIEW WHERE COUNTRY = '"+ a.COUNTRY + "' and INDICATOR = '" +a.indicator+ "'" CMD
	into #GENQUERIES
	from #QUERIES a
		order by a.rowid

if object_id('EDATABLEOUT') is not null drop table EDATABLEOUT
create table EDATABLEOUT (
	 rowid int identity(1, 1)
	,COUNTRY varchar(255)
	,INDICATOR varchar(255)
	,YR int--char(4)
	,AMOUNT float
	)

declare @currow int, @rowcnt int, @sqlcmd varchar(max)
select @currow = 1, @rowcnt = count(*) from #GENQUERIES 
while @currow <= @rowcnt
begin
	select @sqlcmd = cmd from #GENQUERIES where rowid = @currow 
	exec(@sqlcmd)
	select @currow = @currow + 1
end
-- Duration: 1 min 

if object_id('YRS') is not null drop table YRS
select * into YRS from #YRS

if object_id('INDICATORS') is not null drop table INDICATORS
select * into INDICATORS from #INDICATORS

if object_id('COUNTRIES') is not null drop table COUNTRIES
select * into COUNTRIES from #COUNTRIES


