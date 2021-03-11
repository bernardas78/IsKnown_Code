select 
	isnull( (select top 1 ItemName from products p where ic.barcode=p.barcode), name) as ProductName
	, right ( left(barcode, len([Barcode])-5), len([Barcode])-7) as ProductCode --strip 99-XXXXX-00000 as 
	, cnt
from imgcnt ic
--left join products p on ic.barcode=p.barcode
where ic.barcode like '99%00000'	--only weighed products
and cnt>=9
order by --cnt desc,
barcode 
--select barcode,count(*) from products group by barcode having count(*)>1 order by barcode

