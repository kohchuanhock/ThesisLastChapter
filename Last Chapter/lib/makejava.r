#
#
#    RCaller, A solution for calling R from Java
#    Copyright (C) 2010  Mehmet Hakan Satman
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#    Mehmet Hakan Satman - mhsatman@yahoo.com
#    http://www.mhsatman.com
#---------------------------------------------------------------------------

makeNumericArray<-function(obj,javaCode,varname){
  atyp <- "double"
  if (is.integer(obj)) {atyp="int"}
  javaCode<-paste( javaCode,atyp,"[] ",varname,"= new ",atyp,"[]{",sep="" );
  for ( i in 1:length(obj) ) {
	  javaCode<-paste( javaCode,obj[i],sep="" );
    if(i<length(obj)) javaCode<-paste( javaCode,",",sep="" );
  }
  javaCode<-paste( javaCode,"};\n",sep="" );
  return(javaCode);
}

makeAlfaNumericArray<-function(obj,javaCode,varname){
  javaCode<-paste(javaCode,"String[] ",varname,"= new String[]{",sep="");
  for (i in 1:length(obj)){
    javaCode<-paste(javaCode,"\"",obj[i],"\"",sep="");
    if(i<length(obj)) javaCode<-paste(javaCode,",",sep="");
  }
  javaCode<-paste(javaCode,"};\n",sep="");
  return(javaCode);
}

cleanName<-function(name){
  varname<-paste(unlist(strsplit(name,"\\.")),collapse="");
	return(varname);
}

makejava<-function(obj){
	varname<-cleanName(deparse(substitute(obj)));
	javaCode<-"";
	if(is.vector(obj) && is.numeric(obj)){
		javaCode<-makeNumericArray(obj,javaCode,varname);
	}

	if(is.vector(obj) && !is.numeric(obj)){
    javaCode<-makeAlfaNumericArray(obj,javaCode,varname);
  }

	if(is.list(obj)){
		elementNames<-names(obj);
		for (i in 1:length(obj)) {
			if(is.vector(obj[i]) && is.numeric(obj[[i]]) ) {
        javaCode<-makeNumericArray(obj[[i]],javaCode,cleanName(elementNames[i]));
      }
			if(is.vector(obj[i]) && is.character(obj[[i]]) ) {
				javaCode<-makeAlfaNumericArray(obj[[i]],javaCode,cleanName(elementNames[i]));
			}
		}
	}
  return(javaCode);
}