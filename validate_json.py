import json

# Caminho do arquivo corrigido
json_file = 'assets/data/test/test_label_fixed.json'

try:
    # Tenta carregar o arquivo JSON
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Verifica se é um array e quantos elementos tem
    if isinstance(data, list):
        print(f"JSON válido: O arquivo contém um array com {len(data)} elementos.")
        
        # Verifica a estrutura de cada elemento
        errors = 0
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                print(f"Erro no item {i}: não é um objeto JSON")
                errors += 1
                continue
            
            if 'lanes' not in item:
                print(f"Erro no item {i}: chave 'lanes' não encontrada")
                errors += 1
            elif not isinstance(item['lanes'], list):
                print(f"Erro no item {i}: 'lanes' não é uma lista")
                errors += 1
            
            if 'h_samples' not in item:
                print(f"Erro no item {i}: chave 'h_samples' não encontrada")
                errors += 1
            elif not isinstance(item['h_samples'], list):
                print(f"Erro no item {i}: 'h_samples' não é uma lista")
                errors += 1
            
            if 'raw_file' not in item:
                print(f"Erro no item {i}: chave 'raw_file' não encontrada")
                errors += 1
        
        # Mostra detalhes do primeiro elemento para verificação
        if len(data) > 0:
            print("\nExemplo do primeiro elemento:")
            print(f"Número de faixas: {len(data[0]['lanes'])}")
            print(f"Tamanho de h_samples: {len(data[0]['h_samples'])}")
            print(f"raw_file: {data[0]['raw_file']}")
            
        if errors == 0:
            print("\nTodos os elementos têm a estrutura correta!")
        else:
            print(f"\nEncontrados {errors} erros na estrutura dos dados.")
    else:
        print("Erro: O arquivo não contém um array JSON.")
        
except json.JSONDecodeError as e:
    print(f"Erro de decodificação JSON: {e}")
except Exception as e:
    print(f"Erro ao processar o arquivo: {e}") 