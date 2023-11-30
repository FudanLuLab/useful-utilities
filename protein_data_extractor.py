import os, sys
data_file=sys.argv[1]

result_file=open(os.path.join('filename',data_file+'.csv'),'w')
result_file.write(','.join(['Peptide','ratio','protein'])+'\n')

result_dict={}
with open(os.path.join('filename',data_file+'.txt'),'r') as data_file:
    data_file.readline()
    for line in data_file:
        peptide,ratio,gene,protein,ksites=line.strip().split('\t')
        key=','.join([peptide,ratio])
        if key in result_dict:
            result_dict[key].append({protein:':'.join([protein,gene,ksites])})
        else:
            result_dict[key]=[{protein:':'.join([protein,gene,ksites])}]

for key,protein_list in result_dict.items():
    protein_sorted_list=sorted(protein_list,key=lambda x: list(x.keys())[0])
    protein_list=[]
    protein_info_list=[]
    for protein_info_dict in protein_sorted_list:
        protein,protein_info=list(protein_info_dict.items())[0]
        try:
            if protein_list[-1]==protein:
                space_list=['']*(len(protein_list)-1)
                result_file.write(','.join([key,*space_list,protein_info])+'\n')
            else:
                raise Exception()
        except:
            protein_list.append(protein)
            protein_info_list.append(protein_info)
    result_file.write(key+','+','.join(protein_info_list)+'\n')
    