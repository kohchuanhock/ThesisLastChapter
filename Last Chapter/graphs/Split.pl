#!/usr/bin/perl -w
#use strict
        
$^I = ".bak";
while(<>){
	s/Iterations/Bootstrap_replicates/g;
	s/^\s+//;     # replace leading whitespace with nothing
	s/\s+$//;     # replace trailing whitespace with nothing
	if(length($_) > 1000){
		my @fields = split /,/, $_;
		my $count = 1;
		my $string = "";
		foreach $temp (@fields){
			$string .= $temp;
				if($temp =~ /;$/){
					#ends with ;
					$string .= "\n";
				}else{
					$string .= ",";
				}
				
			if($count % 100 == 0){
				print $string."\n";
				$string = "";
				$count = 1;
			}
			$count++;
		}
		print $string."\n";
	}else{
		print $_."\n";
	}
}     
