Filling file_cnts.csv
Get-ChildItem "A:\IsKnown_Images\Cleaned_Aff_NE_AutoVisible\Bal_v14" | Foreach { (Get-ChildItem $_.FullName |  Measure-Object).Count}
Get-ChildItem "A:\IsKnown_Images\Cleaned_Aff_NE_AutoVisible\Bal_v14" | Foreach { echo $_.Name }

Product taxonomy
D:\IsKnown_Code\Insights\categories.xlsx

