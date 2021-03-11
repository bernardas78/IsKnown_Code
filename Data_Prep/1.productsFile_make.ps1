# Make prekes.csv from folder names(barcodes) and file names

# Destination train directory
#$split_folder_pattern = 'A:\RetellectImages\Staging_AllUpTo20201210\2?\*'
$split_folder_pattern = 'A:\RetellectImages\Staging_AllUpTo20210203\2?\*'
#$split_folder_pattern = 'A:\IsKnown_Images\Cleaned\*'

# product file
$MyDir = [System.IO.Path]::GetDirectoryName($myInvocation.MyCommand.Definition)
#echo $MyDir
$prod_file = $MyDir+'\temp\prekes.csv'

# empty out results file
$text | Set-Content $prod_file

# List of found barcodes
#$found_barcodes = New-Object Collections.Generic.List[String]

$found_barcode_file_cnts = New-Object 'system.collections.generic.dictionary[string,int]'
$found_barcode_prod_names = New-Object 'system.collections.generic.dictionary[string,string]'

get-childitem $split_folder_pattern -Directory |
    ForEach-Object {
        #echo $_.FullName
        
        $barcode = $_.FullName.Split('\')[-1]
        $sco = $_.FullName.Split('\')[-2]
        #echo $sco+$barcode

        If (!$found_barcode_file_cnts.Keys.Contains($barcode)) {
            # product name: first filename 
            $filename = (get-childitem $_.FullName | Select -First 1).Name
            #echo $filename
            If ($filename -ne $null){
                #echo ("Again"+$filename)
                # structure date-barcode-product_camid.jpg
                $product_name = $filename.Split("-")[2]
                #echo ("product_name: "+ $barcode + " " + $product_name)
                $found_barcode_prod_names[$barcode] = $product_name
            }
            If ($found_barcode_prod_names[$barcode] -eq $null -Or $found_barcode_prod_names[$barcode] -eq "") {
                $found_barcode_prod_names[$barcode] = $barcode
            }
        }

        # file count
        $filecount = (get-childitem $_.FullName).Count
        $found_barcode_file_cnts[$barcode] += $filecount
        #echo $filename
        
        #$product_name + "," + $barcode | Add-Content $prod_file
    }

$found_barcode_file_cnts.Keys.GetEnumerator() | Sort |
    ForEach-Object {
        #echo $_
        #echo $found_barcode_file_cnts[$_]
        #echo $found_barcode_prod_names[$_]
        $found_barcode_prod_names[$_] + "," + $_ + "," + $found_barcode_file_cnts[$_] | Add-Content $prod_file
    }