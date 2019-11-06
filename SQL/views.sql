/*
Author: Alan Danque
Date:	20191018
Purpose:Create data focused views
*/

create view vwPopulationTotal_vs_LaborForceTotal
as
select sub1.COUNTRY, sub1.Democratic, sub1.YR, sub1.[Population total], sub2.[Labor force total]
from
	(
	select a.COUNTRY, a.Democratic,  a.YR, a.AMOUNT as 'Population total' 
		from EDATABLEOUT a where indicator = 'Population total' --1927
	) sub1
	join 
		(
		select a.COUNTRY, a.Democratic,  a.YR, a.AMOUNT as 'Labor force total' 
			from EDATABLEOUT a where indicator = 'Labor force total' --1927
		) sub2
			on sub1.COUNTRY = sub2.COUNTRY
				and sub1.YR = sub2.YR
		where sub1.YR between 1999 and 2014
go


create view vwPopulationTotal_vs_LaborForceTotal
as
select sub1.COUNTRY, sub1.Democratic, sub1.YR, sub1.[Population total], sub2.[Labor force total]
from
	(
	select a.COUNTRY, a.Democratic,  a.YR, a.AMOUNT as 'Population total' 
		from EDATABLEOUT a where indicator = 'Population total' --1927
	) sub1
	join 
		(
		select a.COUNTRY,a.Democratic,  a.YR, a.AMOUNT as 'Labor force total' 
			from EDATABLEOUT a where indicator = 'Labor force total' --1927
		) sub2
			on sub1.COUNTRY = sub2.COUNTRY
				and sub1.YR = sub2.YR
		where sub1.YR between 1999 and 2014
go


create view vwLabor_force_total_BY_Population_total_vs_GDP_per_capita_current_US
as
select sub1.COUNTRY, sub1.Democratic, sub1.YR, sub1.[Labor force total]/sub2.[Population total] LaborToPopulationPct, sub3.[GDP per capita current US]
from
	(
	select a.COUNTRY,a.Democratic,  a.YR, a.AMOUNT as 'Labor force total' 
		from EDATABLEOUT a where indicator = 'Labor force total' --1927
	) sub1
	join 
		(
		select a.COUNTRY, a.Democratic,  a.YR, a.AMOUNT as 'Population total' 
			from EDATABLEOUT a where indicator = 'Population total' --1927
		) sub2
			on sub1.COUNTRY = sub2.COUNTRY
				and sub1.YR = sub2.YR
	join 
		(
		select a.COUNTRY, a.Democratic, a.YR, a.AMOUNT as 'GDP per capita current US' 
			from EDATABLEOUT a where indicator = 'GDP per capita current US' --1927
		) sub3
			on sub1.COUNTRY = sub3.COUNTRY
				and sub1.YR = sub3.YR
	where sub1.YR between 1999 and 2014
go


create view vwGovernment_expenditure_per_tertiary_student_US_vs_GDP_per_capita_current_US
as
select sub1.COUNTRY, sub1.Democratic, sub1.YR, sub1.[Government expenditure per tertiary student US], sub2.[GDP per capita current US]
from
	(
	select a.COUNTRY ,a.Democratic, a.YR, a.AMOUNT as 'Government expenditure per tertiary student US' 
		from EDATABLEOUT a where indicator = 'Government expenditure per tertiary student US' --1927
	) sub1
	join 
		(
		select a.COUNTRY,a.Democratic,  a.YR, a.AMOUNT as 'GDP per capita current US' 
			from EDATABLEOUT a where indicator = 'GDP per capita current US' --1927
		) sub2
			on sub1.COUNTRY = sub2.COUNTRY
				and sub1.YR = sub2.YR
		where sub1.YR between 1999 and 2014
go


create view vwGovernment_expenditure_per_tertiary_student_US_vs_Labor_force_total
as
select sub1.COUNTRY, sub1.Democratic, sub1.YR, sub1.[Government expenditure per tertiary student US], sub2.[Labor force total]
from
	(
	select a.COUNTRY, a.Democratic, a.YR, a.AMOUNT as 'Government expenditure per tertiary student US' 
		from EDATABLEOUT a where indicator = 'Government expenditure per tertiary student US' --1927
	) sub1
	join 
		(
		select a.COUNTRY, a.Democratic, a.YR, a.AMOUNT as 'Labor force total' 
			from EDATABLEOUT a where indicator = 'Labor force total' --1927
		) sub2
			on sub1.COUNTRY = sub2.COUNTRY
				and sub1.YR = sub2.YR
		where sub1.YR between 1999 and 2014
go


create view vwRegressionAnalysisDataset
as
select sub1.COUNTRY
	, sub1.Democratic
	, sub1.YR
	, sub1.[Labor force total]/sub2.[Population total] LaborToPopulationPct
	, sub3.[GDP per capita current US] GDP
	, dbo.udf_PctValues(sub3.[GDP per capita current US]) GDPRNK
	, sub1.[Labor force total]  LABORFORCE
	, sub2.[Population total]  POPULATIONTOTAL
	, sub4.[TertiaryExpenditure]  TertiaryExpenditure
from
	(
	select a.COUNTRY,a.Democratic,  a.YR, a.AMOUNT as 'Labor force total' 
		from EDATABLEOUT a where indicator = 'Labor force total' --1927
	) sub1
	join 
		(
		select a.COUNTRY, a.Democratic,  a.YR, a.AMOUNT as 'Population total' 
			from EDATABLEOUT a where indicator = 'Population total' --1927
		) sub2
			on sub1.COUNTRY = sub2.COUNTRY
				and sub1.YR = sub2.YR
	join 
		(
		select a.COUNTRY, a.Democratic, a.YR, a.AMOUNT as 'GDP per capita current US' 
			from EDATABLEOUT a where indicator = 'GDP per capita current US' --1927
		) sub3
			on sub1.COUNTRY = sub3.COUNTRY
				and sub1.YR = sub3.YR
	join 
		(
		select a.COUNTRY, a.Democratic, a.YR, a.AMOUNT as 'TertiaryExpenditure' 
				from EDATABLEOUT a where indicator = 'Government expenditure per tertiary student US' --1927
		) sub4
			on sub1.COUNTRY = sub4.COUNTRY
				and sub1.YR = sub4.YR
	where sub1.YR between 1999 and 2014
		and sub4.[TertiaryExpenditure] > 0 
		and sub1.[Labor force total] > 0 
		and sub2.[Population total] > 0
go


