# Relatório de Classificação de Texto

## a) Definição e Apresentação da Estrutura e Divisão do Dataset para Treinamento, Validação e Teste

Este projeto envolve a classificação de transcrições médicas de várias especialidades. Devido a limitações de processamento, foi usada uma amostra representativa dos dados.
![Distribuição das classes no dataset completo](https://github.com/user-attachments/assets/e7559b47-be8a-455a-bdfb-3ce3602decbf)

A imagem acima mostra a distribuição de classes no dataset completo.

![Amostra utilizada para treinamento e validação](https://github.com/user-attachments/assets/82ade9f4-5d43-4195-b06e-77178cec0300)
A imagem acima ilustra uma amostra representativa do dataset que foi utilizada para o treinamento e validação do modelo.

## b) Apresentação dos Atributos que Compõem o Dataset

O dataset inclui as seguintes colunas: especialidade médica e transcrição. A transcrição contém o texto médico que será utilizado para a classificação.

Link para o dataset: https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions

![Informações dos atributos do dataset](https://github.com/user-attachments/assets/b6ddf7ba-8c5c-4613-8970-5710992236b1)

A figura acima detalha as informações sobre os atributos do dataset, incluindo as opções de classes para a classificação.

## c) Apresentação dos Primeiros Cinco Registros do Dataset

Para entender melhor o conteúdo do dataset, apresentamos os primeiros cinco registros com suas respectivas classes associadas.
![Cinco primeiros registros do dataset](https://github.com/user-attachments/assets/730ca3b1-fd8e-41c8-b443-3922f7b17193)

A imagem acima mostra os primeiros cinco registros do dataset, fornecendo um exemplo das transcrições e suas respectivas classes.

## d) Distribuição das Classes ao Longo dos Registros

Para visualizar como as classes estão distribuídas no dataset, apresentamos um gráfico que mostra a quantidade de registros por classe.
![Distribuição das classes no dataset](https://github.com/user-attachments/assets/6065cdb9-5151-4b2e-9d6f-5d47b6ee1e30)

O gráfico acima ilustra a distribuição das classes ao longo dos registros no dataset.

## e) Tamanho dos Registros por Cada Classe

A seguir, apresentamos um box-plot para visualizar o tamanho dos registros por classe. Esse gráfico ajuda a entender a variabilidade no comprimento das transcrições para cada classe.

![Tamanho dos registros por classe](https://github.com/user-attachments/assets/ee636c33-53f6-45b1-88f4-c6269f426b16)

O box-plot acima exibe a distribuição do tamanho dos registros para cada classe, permitindo identificar padrões e variações no comprimento das transcrições.

## f) Criação dos Tokens do Dataset

Para o pré-processamento dos dados, utilizamos a classe `AutoTokenizer` para tokenizar as transcrições médicas.

```python
# Carregar o tokenizer do modelo pré-treinado
tokenizer = AutoTokenizer.from_pretrained('finiteautomata/bertweet-base-sentiment-analysis')

# Função para tokenizar o texto
def tokenize_text(text):
    return tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors='pt')

# Aplicar tokenização ao dataset
filtered_clinical_text['tokens'] = filtered_clinical_text['transcription'].apply(tokenize_text)

# Verificar a estrutura dos tokens
print("Estrutura dos tokens:")
print(filtered_clinical_text['tokens'].head())

# Exibir exemplos de tokens
print("\nExemplos de tokens:")
for i, tokens in enumerate(filtered_clinical_text['tokens'].head()):
    print(f"\nTexto original: {filtered_clinical_text['transcription'].iloc[i]}")
    print(f"Tokens: {tokens['input_ids'].squeeze().tolist()}")
    print(f"Tokens decodificados: {tokenizer.convert_ids_to_tokens(tokens['input_ids'].squeeze().tolist())}")

# Verificar o comprimento dos tokens
print("\nComprimento dos tokens:")
filtered_clinical_text['length_of_tokens'] = filtered_clinical_text['tokens'].apply(lambda x: x['input_ids'].shape[1])
print(filtered_clinical_text[['medical_specialty', 'length_of_tokens']].head())
```

O código acima usa um modelo pré-treinado para criar tokens a partir das transcrições.

## g) Geração dos Embeddings Resultantes dos Estados Escondidos

Após a tokenização, utilizamos o modelo pré-treinado para gerar embeddings a partir dos tokens.

### Descrição do Modelo

`finiteautomata/bertweet-base-sentiment-analysis` é um modelo de classificação de sentimentos treinado com o corpus SemEval 2017, que contém aproximadamente 40.000 tweets. O modelo é ajustado para realizar análise de sentimentos, classificando os textos em três categorias: POS (positivo), NEG (negativo) e NEU (neutro).

- **Base do Modelo**: `BERTweet`, uma variante do `RoBERTa`.
- **Treinamento**: O modelo foi treinado utilizando o corpus SemEval 2017.
- **Classes de Saída**: O modelo classifica os textos em três rótulos - POS, NEG e NEU.
- **Licença**: O uso do modelo é restrito a fins não comerciais e de pesquisa científica.

## h) Extração e Apresentação dos Últimos Estados Escondidos

Os últimos estados escondidos são extraídos e apresentados para análise.

Shape do último estado escondido: `torch.Size([1, 128, 768])`

Podemos ver que é um vetor de 768 dimensões.

## i) Conversão dos Estados Escondidos para Tensores PyTorch

Convertendo os estados escondidos para tensores PyTorch, preparamos os dados para o treinamento do modelo.

```python
import torch

train_embeddings = torch.tensor(last_hidden_states)
```

## j) Estrutura do Tensor

Para entender melhor os dados, mostramos a estrutura do tensor gerado.

O shape `torch.Size([1, 128, 768])` dos últimos estados escondidos de um modelo de linguagem, como o BERTweet, pode ser interpretado da seguinte forma:

- **Batch Size**: O primeiro valor 1 indica o tamanho do lote (batch size) de entrada. Nesse caso, significa que apenas um exemplo (ou uma sequência) foi processado por vez.
- **Sequência**: O segundo valor 128 representa o comprimento da sequência ou o número de tokens na entrada. Isso indica que a sequência de entrada foi truncada ou padronizada para ter 128 tokens. Esse comprimento pode variar dependendo da configuração do modelo e do pré-processamento.
- **Dimensão dos Estados Ocultos**: O terceiro valor 768 refere-se à dimensão dos estados ocultos ou a dimensão dos embeddings gerados pelo modelo. No caso do BERTweet, essa dimensão é baseada no tamanho do embedding da camada final do modelo. Para BERT e suas variantes, a dimensão dos estados ocultos é geralmente 768 para o modelo base.

## k) Criação dos Vetores de Treinamento e Validação

Criamos vetores de treinamento e validação e exibimos seus formatos.

```python
X_train_features = train_embeddings.view(train_embeddings.size(0), -1).numpy()
X_val_features = val_embeddings.view(val_embeddings.size(0), -1).numpy()

print("Formato do vetor de treinamento:", X_train_features.shape)
print("Formato do vetor de validação:", X_val_features.shape)
```

## l) Visualização do Conjunto de Treinamento com UMAP

A visualização UMAP reduz as dimensões dos dados para facilitar a análise.

![visualização UMAP](https://github.com/user-attachments/assets/42aceafb-d4ff-40ca-bf99-9a2ec06b7c87)

## m) Interpretação dos Resultados

O gráfico resultante da visualização dos embeddings demonstra que as classes, em grande parte, estão separadas no espaço de duas dimensões, o que indica que o modelo conseguiu aprender representações que distinguem razoavelmente bem as diferentes especialidades médicas. Isso sugere que os textos relacionados a áreas como "Neurologia" e "Ortopedia" possuem características únicas que os diferenciam dos textos de outras especialidades.

No entanto, também é evidente uma sobreposição significativa entre algumas classes, o que indica que o modelo enfrenta dificuldades para distinguir entre elas.

## n) Treinamento com Regressão Logística

Treinamos um classificador de regressão logística e apresentamos a acurácia obtida.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_features, y_train)
y_pred = clf.predict(X_val_features)
accuracy = accuracy_score(y_val, y_pred)

print(f'Accuracy of Logistic Regression: {accuracy:.4f}')
```

Accuracy of Logistic Regression: `0.4500`

## o) Comparação com DummyClassifier

Comparamos a acurácia do classificador de regressão logística com a do DummyClassifier.

```python
from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train_features, y_train)
y_pred_dummy = dummy_clf.predict(X_val_features)
accuracy_dummy = accuracy_score(y_val, y_pred_dummy)

print(f'Accuracy of Dummy Classifier: {accuracy_dummy:.4f}')
```

Accuracy of DummyClassifier: `0.1900`

Accuracy of Logistic Regression: `0.4500`

## p) Apresentação da Matriz de Confusão

Apresentamos a matriz de confusão da regressão logística e discutimos os resultados.
![Matriz de confusão](https://github.com/user-attachments/assets/5dca0641-5a14-4c83-9420-b83cf8f04b64)

```python
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

cm = confusion_matrix(y_val, y_pred, labels=clf.classes_)
fig, ax = plt.subplots(figsize=(8, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap='Blues', values_format='.2f', ax=ax, colorbar=False)
plt.title("Matriz de Confusão - Regressão Logística")
plt.show()
```

A matriz de confusão ajuda a visualizar como as previsões do modelo estão distribuídas em relação às classes reais. Observamos que o modelo tem uma maior taxa de acertos para certas classes, mas ainda há confusões, especialmente entre especialidades com transcrições mais semelhantes.

## r) Conclusões e Próximos Passos

O projeto de classificação de transcrições médicas utilizando um modelo de linguagem pré-treinado como o BERTweet demonstrou resultados promissores. A acurácia obtida com o classificador de regressão logística é significativamente maior do que a obtida com o DummyClassifier, o que indica que o modelo é capaz de capturar padrões úteis nos dados.

No entanto, ainda há espaço para melhorias, especialmente na distinção entre classes semelhantes. Para futuras iterações, consideraremos:

1. **Ajuste Fino do Modelo**: Refinar o modelo pré-treinado com um corpus específico de transcrições médicas pode melhorar o desempenho.
2. **Exploração de Outras Técnicas de Pré-Processamento**: Técnicas adicionais de pré-processamento, como a utilização de embeddings mais avançados ou o uso de abordagens de data augmentation, podem ser exploradas.
3. **Avaliação de Outros Modelos**: Testar outros modelos de classificação, como redes neurais profundas ou SVMs, pode proporcionar melhorias.

Este relatório fornece uma base sólida para o desenvolvimento contínuo de modelos de NLP para a classificação de textos médicos, com o objetivo final de melhorar a precisão e a eficácia dessas ferramentas em aplicações práticas.
