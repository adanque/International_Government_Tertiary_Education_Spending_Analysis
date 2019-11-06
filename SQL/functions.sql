
create function dbo.udf_PctValues (@Value float)
returns float
as
/*
Author: Alan Danque
Date:	20191020
Purpose:Dynamically creates a rank
*/
begin
	declare 
		 @5th float
		,@10th float
		,@15th float
		,@20th float
		,@25th float
		,@30th float
		,@35th float
		,@40th float
		,@45th float
		,@50th float
		,@55th float
		,@60th float
		,@65th float
		,@70th float
		,@75th float
		,@80th float
		,@85th float
		,@90th float
		,@95th float
		,@100th float
		,@RetValue float


	select 
		 @5th = round((max(gdp)/20)+(min(gdp)/20), 0, 0) 
		,@10th = round(((max(gdp)/20)+(min(gdp)/20))*2, 0, 0) 
		,@15th = round(((max(gdp)/20)+(min(gdp)/20))*3, 0, 0) 
		,@20th = round(((max(gdp)/20)+(min(gdp)/20))*4, 0, 0) 
		,@25th = round(((max(gdp)/20)+(min(gdp)/20))*5, 0, 0) 
		,@30th = round(((max(gdp)/20)+(min(gdp)/20))*6, 0, 0) 
		,@35th = round(((max(gdp)/20)+(min(gdp)/20))*7, 0, 0) 
		,@40th = round(((max(gdp)/20)+(min(gdp)/20))*8, 0, 0) 
		,@45th = round(((max(gdp)/20)+(min(gdp)/20))*9, 0, 0) 
		,@50th = round(((max(gdp)/20)+(min(gdp)/20))*9, 0, 0) 
		,@55th = round(((max(gdp)/20)+(min(gdp)/20))*10, 0, 0) 
		,@60th = round(((max(gdp)/20)+(min(gdp)/20))*12, 0, 0) 
		,@65th = round(((max(gdp)/20)+(min(gdp)/20))*13, 0, 0) 
		,@70th = round(((max(gdp)/20)+(min(gdp)/20))*14, 0, 0) 
		,@75th = round(((max(gdp)/20)+(min(gdp)/20))*15, 0, 0) 
		,@80th = round(((max(gdp)/20)+(min(gdp)/20))*16, 0, 0) 
		,@85th = round(((max(gdp)/20)+(min(gdp)/20))*17, 0, 0) 
		,@90th = round(((max(gdp)/20)+(min(gdp)/20))*18, 0, 0) 
		,@95th = round(((max(gdp)/20)+(min(gdp)/20))*19, 0, 0) 
		,@100th = round(max(gdp), 0, 0) 
	from vwRegressionAnalysisDataset

	select @RetValue = case 
							when @value < @5th then @5th
							when @value < @10th then @10th
							when @value < @15th then @15th
							when @value < @20th then @20th
							when @value < @25th then @25th
							when @value < @30th then @30th
							when @value < @35th then @35th
							when @value < @40th then @40th
							when @value < @45th then @45th
							when @value < @50th then @50th
							when @value < @55th then @55th
							when @value < @60th then @60th
							when @value < @65th then @65th
							when @value < @70th then @70th
							when @value < @75th then @75th
							when @value < @80th then @80th
							when @value < @85th then @85th
							when @value < @90th then @90th
							when @value < @95th then @95th
							else @100th
						end
	return(@RetValue);
end;
go

