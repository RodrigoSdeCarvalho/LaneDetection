import json
import os

# Caminho do arquivo original e do novo arquivo
input_file = 'assets/data/test/test_label_new.json'
output_file = 'assets/data/test/test_label_fixed.json'

# Verifica se o arquivo de saída já existe e remove se necessário
if os.path.exists(output_file):
    os.remove(output_file)

# Abre o arquivo de saída e escreve o início do array
with open(output_file, 'w') as out_file:
    out_file.write('[\n')

# Lê o arquivo linha por linha e converte cada objeto JSON separadamente
with open(input_file, 'r') as in_file:
    lines = in_file.readlines()
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        try:
            # Verifica se a linha contém um objeto JSON válido
            json_obj = json.loads(line)
            
            # Escreve o objeto no arquivo de saída
            with open(output_file, 'a') as out_file:
                if i < len(lines) - 1 and lines[i+1].strip():
                    out_file.write(line + ',\n')
                else:
                    out_file.write(line + '\n')
        except json.JSONDecodeError:
            print(f"Erro na linha {i+1}: {line[:50]}...")

# Fecha o array JSON
with open(output_file, 'a') as out_file:
    out_file.write(']\n')

print(f"Arquivo corrigido salvo em: {output_file}") 