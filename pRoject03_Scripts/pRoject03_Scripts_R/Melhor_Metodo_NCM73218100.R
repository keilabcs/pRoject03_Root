#### Projeto3

##### CARREGANDO PACOTES

install.packages("tm")
install.packages("SnowballC")
install.packages("wordcloud")

install.packages("igraph") #Grafo 
install.packages("NLP") # ngrams

install.packages("readr") #tokenize
install.packages("utils") #View

install.packages("factoextra") # Install factoextra
install.packages("cluster") # Install cluster package

install.packages("preText") #PreText 2
install.packages("ggraph")
install.packages("widyr")
install.packages("tidytext")
install.packages("url")
install.packages("curl")

##### LENDO BIBLIOTECAS

library(curl)
library(tm)
library(SnowballC)
library(wordcloud)
library(NLP)
library(igraph)

library(cluster)
library(factoextra)

library(readr) #tokenize
library(utils) #View

library(preText)
library(quanteda)

library(dplyr)
library(tidytext)
library(janeaustenr)
library(tidyr)

library(ggraph)
library(url)
##############################################

#Ler dados
X1 <- UFAL_2016_TODOS_SEM_84219999_REVISADO_CLASSIFICADA
X1

#Definir NCM
B <- X1$DESCR_PROD[X1$NCM_PROD == 73218100] #DELIMITAR POR NCM
B

aux <- B
auxCorpus <- Corpus(VectorSource(aux)) #criar um corpus

documents <- auxCorpus

print(names(documents)) # Dê uma olhada nos nomes dos documentos

preprocessed_documents <- factorial_preprocessing(
  documents,
  use_ngrams = TRUE,
  infrequent_term_threshold = 0.06,
  verbose = TRUE)
## Preprocessing os documentos 128 maneiras diferentes ...
## As principais funções irão pré-processar o texto de entrada 64-128 maneiras diferentes
#use_ngrams = TRUE definimos o limite de proporção do documento para remover os termos infreqüentes em 0,5. 
#Isso significa que termos que aparecem em menos de 10 por cento (2/10) documentos serão removidos.

names(preprocessed_documents)

## [1] "choices"  "dfm_list" "labels"  
head(preprocessed_documents$choices) #Frame de dados contendo indicadores para cada uma das etapas de pré-processamento usadas.
head(preprocessed_documents$dfm_list) #lista com 64 ou 128 entradas.
head(preprocessed_documents$labels)

#Selecionamos 50 Comparações por pares para fins ilustrativos.
preText_results <- preText(
  preprocessed_documents,
  dataset_name = "NCM PROD 73218100",
  distance_method = "cosine",
  num_comparisons = 150,
  verbose = TRUE)

preText_score_plot(preText_results) #gráfico de pontos de pontuações para cada especificação de pré-processamento.
#As especificações menos "arriscadas" têm a menor pontuação de pré-texto e são exibidas na parte superior do gráfico.

#R^2 qto mais perto de 1 melhor!
#The R^2 for this model is: 0.8923269
#89,23% da variável dependente consegue ser explicada pelos regressores presentes no modelo.

#Resultados de regressão (os coeficientes negativos implicam menos risco).
#Variable Coefficient    SE
#1               Intercept       0.084 0.003
#2      Remove Punctuation      -0.001 0.002
#3          Remove Numbers       0.002 0.002
#4               Lowercase       0.067 0.002
#5                Stemming       0.000 0.002
#6        Remove Stopwords       0.000 0.002
#7 Remove Infrequent Terms       0.027 0.002
#8              Use NGrams      -0.008 0.002


#Coeficiente negativo indica que um passo tende a reduzir o incomum dos resultados, 
#enquanto um coeficiente positivo indica que a aplicação do passo é susceptível de 
#produzir resultados mais "incomuns" para esse corpus.
#Um modelo de n- gramas é um tipo de modelo de linguagem probabilística para prever o 
#próximo item em tal seqüência na forma de um modelo de Markov de ordem ( n  - 1) .
regression_coefficient_plot(preText_results,
                            remove_intercept = TRUE)

################################ testa o Metodo

text <- B
text 

#Transformá-lo em um conjunto de dados de texto arrumado
text_df <- data_frame(line = 1:30, text = text)
text_df

#Converter isso para que ele tenha um-token-por-documento-por-linha
text_df %>%
  unnest_tokens(word, text)

text_df 

#Tokenizing por n-grama
austen_bigrams <- text_df %>%
  unnest_tokens(bigram, text, token = "ngrams", n = 2)

austen_bigrams

#Análise de n-grama
austen_bigrams %>%
  count(bigram, sort = TRUE)

#remover casos em que qualquer stop-word.
bigrams_separated <- austen_bigrams %>%
  separate(bigram, c("word1", "word2"), sep = " ")

bigrams_filtered <- bigrams_separated %>%
  filter(!word1 %in% stop_words$word) %>%
  filter(!word2 %in% stop_words$word)

# new bigram counts:
bigram_counts <- bigrams_filtered %>% 
  count(word1, word2, sort = TRUE)

bigram_counts


#Bigram mais comuns s stopwords
bigrams_united <- bigrams_filtered %>%
  unite(bigram, word1, word2, sep = " ")

bigrams_united

#trigramas mais comuns

austen_books <- text_df %>%
  unnest_tokens(trigram, text, token = "ngrams", n = 3) %>%
  separate(trigram, c("word1", "word2", "word3"), sep = " ") %>%
  filter(!word1 %in% stop_words$word,
         !word2 %in% stop_words$word,
         !word3 %in% stop_words$word) %>%
count(word1, word2, word3, sort = TRUE)

austen_books

#podemos olhar para o tf-idf ***

bigram_tf_idf <- bigrams_united %>%
  count(book, bigram) %>%
  bind_tf_idf(bigram, book, n) %>%
  arrange(desc(tf_idf))

bigram_tf_idf

#Visualizando uma rede de bigrams com ggraph

bigram_counts

bigram_graph <- bigram_counts %>%
  filter(n > 1) %>%
  graph_from_data_frame()

bigram_graph

set.seed(2017)
ggraph(bigram_graph, layout = "fr") +
  geom_edge_link() +
  geom_node_point() +
  geom_node_text(aes(label = name), vjust = 1, hjust = 1)

#Adicionando Tema ao Bigram
#visualização de uma cadeia de Markov , um modelo comum no processamento de texto.
#Em uma cadeia de Markov, cada escolha de palavra depende apenas da palavra anterior.
set.seed(2016)

a <- grid::arrow(type = "closed", length = unit(.15, "inches"))

ggraph(bigram_graph, layout = "fr") +
  geom_edge_link(aes(edge_alpha = n), show.legend = FALSE,
                 arrow = a, end_cap = circle(.07, 'inches')) +
  geom_node_point(color = "lightblue", size = 5) +
  geom_node_text(aes(label = name), vjust = 1, hjust = 1) +
  theme_void()


