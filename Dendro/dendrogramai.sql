select 
	isnull( (select top 1 ItemName from products p where ic.barcode=p.barcode), name) as ProductName
	, right ( left(barcode, len([Barcode])-5), len([Barcode])-7) as ProductCode --strip 99-XXXXX-00000 as 
	, cnt
from imgcnt ic
--left join products p on ic.barcode=p.barcode
where ic.barcode like '99%00000'	--only weighed products
and cnt>3
order by --cnt desc,
barcode 
--select barcode,count(*) from products group by barcode having count(*)>1 order by barcode


;with indProds as
(select 
	isnull( (select top 1 ItemName from products p where ic.barcode=p.barcode), name) as ProductName
	, right ( left(barcode, len([Barcode])-5), len([Barcode])-7) as ProductCode --strip 99-XXXXX-00000 as 
	, cnt
from imgcnt ic
where ic.barcode like '99%00000'	--only weighed products
and cnt>3)
select left (ProductCode,4) as ProductCode, sum(cnt) as Cnt
from indProds
group by left (ProductCode,4)
order by left (ProductCode,4)


;with indProds as
(select 
	isnull( (select top 1 ItemName from products p where ic.barcode=p.barcode), name) as ProductName
	, right ( left(barcode, len([Barcode])-5), len([Barcode])-7) as ProductCode --strip 99-XXXXX-00000 as 
	, cnt
from imgcnt ic
where ic.barcode like '99%00000'	--only weighed products
and cnt>3)
select left (ProductCode,3) as ProductCode, sum(cnt) as Cnt
from indProds
group by left (ProductCode,3)
order by left (ProductCode,3)

;with indProds as
(select 
	isnull( (select top 1 ItemName from products p where ic.barcode=p.barcode), name) as ProductName
	, right ( left(barcode, len([Barcode])-5), len([Barcode])-7) as ProductCode --strip 99-XXXXX-00000 as 
	, cnt
from imgcnt ic
where ic.barcode like '99%00000'	--only weighed products
and cnt>3)
select left (ProductCode,2) as ProductCode, sum(cnt) as Cnt
from indProds
group by left (ProductCode,2)
order by left (ProductCode,2)

;with indProds as
(select 
	isnull( (select top 1 ItemName from products p where ic.barcode=p.barcode), name) as ProductName
	, right ( left(barcode, len([Barcode])-5), len([Barcode])-7) as ProductCode --strip 99-XXXXX-00000 as 
	, cnt
from imgcnt ic
where ic.barcode like '99%00000'	--only weighed products
and cnt>3)
select left (ProductCode,1) as ProductCode, sum(cnt) as Cnt
from indProds
group by left (ProductCode,1)
order by left (ProductCode,1)