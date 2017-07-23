### pRoject 03

##### CARREGANDO PACOTES

install.packages("NLP") # ngrams
library(NLP)

install.packages("tm")
library(tm)

install.packages("SnowballC")
library(SnowballC)

install.packages("wordcloud2")
library(wordcloud2)

install.packages("igraph") #Grafo 
library(igraph)

install.packages("readr") #tokenize
library(readr) #tokenize

install.packages("utils") #View
library(utils) #View

install.packages("ggplot2")
library(ggplot2)

install.packages("factoextra") # Install factoextra
library(factoextra)

install.packages("cluster") # Install cluster package
library(cluster)

install.packages("ggraph")
library(ggraph)

install.packages("dplyr")
library(dplyr)

install.packages("tidytext")
library(tidytext)

install.packages("slam")
library("slam")

install.packages("preText") #PreText 2
library(preText)


##### LENDO BIBLIOTECAS


library(quanteda)
library(janeaustenr)
library(tidyr)


############################################## NCM 73211100

#Ler dados
X1 <- UFAL_2016_TODOS_SEM_84219999_REVISADO_CLASSIFICADA
X1

#Definir NCM
B <- X1$DESCR_PROD[X1$NCM_PROD == 73211100] #DELIMITAR POR NCM
B

##### Iniciando com o PreText

#Matthew J. Denny, and Arthur Spirling (2017). “Text Preprocessing For Unsupervised Learning: Why It Matters, When It Misleads, And What To Do About It”.

########################### Verificar Melhor metodo p conj. de dados
#Um pacote R para avaliar os efeitos das decisões de pré-processamento de texto.

#https://cran.r-project.org/web/packages/preText/vignettes/getting_started_with_preText.html

aux <- B
auxCorpus <- Corpus(VectorSource(aux)) #criar um corpus

documents1 <- auxCorpus

print(names(documents1)) # Dê uma olhada nos nomes dos documentos

#[1] "1"    "2"    "3"    "4"    "5"    "6"    "7"    "8"    "9"    "10"   "11"   "12"   "13"   "14"   "15"   "16"  
#[17] "17"   "18"   "19"   "20"   "21"   "22"   "23"   "24"   "25"   "26"   "27"   "28"   "29"   "30"   "31"   "32"  
#[33] "33"   "34"   "35"   "36"   "37"   "38"   "39"   "40"   "41"   "42"   "43"   "44"   "45"   "46"   "47"   "48"  
#[49] "49"   "50"   "51"   "52"   "53"   "54"   "55"   "56"   "57"   "58"   "59"   "60"   "61"   "62"   "63"   "64"  

#A factorial_preprocessing()função, que irá processar os dados 64 ou 128 maneiras diferentes (dependendo se os n-gramas estão incluídos).
## Preprocessing os documentos 128 maneiras diferentes ...
#É altamente desaconselhável usar mais de 500-1000 sob quaisquer circunstâncias e no caso em que o usuário deseja 
#processar mais de algumas centenas de documentos, eles podem querer explorar a parallelopção. 
#Isso pode acelerar significativamente o pré-processamento, mas exigirá significativamente mais RAM no computador que está sendo usado.
#use_ngrams = TRUE definimos o limite de proporção do documento para remover os termos infreqüentes em 0,2. 
#O valor padrão é 0,01 (ou 1/100 documentos)
#Isso significa que termos que aparecem em menos de (2/10) documentos serão removidos.

preprocessed_documents1 <- factorial_preprocessing(
  documents1,
  use_ngrams = TRUE,
  infrequent_term_threshold = 0.2,
  verbose = TRUE)
  

names(preprocessed_documents1)

#Esta função exibirá um objeto de lista com três campos. O primeiro deles é ($choicesum) quadro de dados contendo 
#indicadores para cada uma das etapas de pré-processamento usadas. A segunda é $dfm_list, que é uma lista com 64 ou 128
#entradas, cada uma das quais contém um quanteda::dfmobjeto pré-processado de acordo com a especificação na linha 
#correspondente choices. Cada DFM nesta lista será rotulado para coincidir com os nomes das filas em escolhas, mas 
#você também pode acessar esses rótulos a partir do $labelscampo. Podemos observar as primeiras linhas choicesabaixo:

## [1] "choices"  "dfm_list" "labels"  


head(preprocessed_documents1$choices) #Frame de dados contendo indicadores para cada uma das etapas de pré-processamento usadas.
                   #removePunctuation removeNumbers lowercase(Minuscula)    stem(redução do vocabulário)    removeStopwords infrequent_terms use_ngrams
#P-N-L-S-W-I-3              TRUE          TRUE      TRUE                     TRUE                           TRUE             TRUE            TRUE
#N-L-S-W-I-3               FALSE          TRUE      TRUE                     TRUE                           TRUE             TRUE            TRUE
#P-L-S-W-I-3                TRUE         FALSE      TRUE                     TRUE                           TRUE             TRUE            TRUE
#L-S-W-I-3                 FALSE         FALSE      TRUE                     TRUE                           TRUE             TRUE            TRUE
#P-N-S-W-I-3                TRUE          TRUE     FALSE                     TRUE                           TRUE             TRUE            TRUE
#N-S-W-I-3                 FALSE          TRUE     FALSE                     TRUE                           TRUE             TRUE            TRUE

head(preprocessed_documents1$dfm_list) #lista com 64 ou 128 entradas.
#$`P-N-L-S-W-I-3`
#Document-feature matrix of: 12,134 documents, 6 features (61% sparse).
#$`N-L-S-W-I-3`
#Document-feature matrix of: 12,134 documents, 7 features (62.5% sparse).
#$`P-L-S-W-I-3`
#Document-feature matrix of: 12,134 documents, 6 features (61% sparse).

head(preprocessed_documents1$labels)
#[1] "P-N-L-S-W-I-3" "N-L-S-W-I-3"   "P-L-S-W-I-3"   "L-S-W-I-3"     "P-N-S-W-I-3"   "N-S-W-I-3"


#Com os documentos pré-processados, podemos executar o procedimento preText no corpus factorial pré-processado 
#usando a preText()função. Será útil agora dar um nome ao nosso conjunto de dados usando o dataset_name argumento, 
#pois isso aparecerá em alguns dos gráficos que geramos com a saída. O número padrão de pares a comparar é de 50 para 
#corpos de tamanho razoável.

#O número máximo de distâncias do documento em pares é apenas (20) * (20 - 1) / 2 = 190, 
#então selecionamos 20 Comparações por pares para fins de ilustração.

#Selecionar 20 Comparações por pares para fins ilustrativos.
preText_results <- preText(
  preprocessed_documents1,
  dataset_name = "NCM PROD 73211100",
  distance_method = "cosine",
  num_comparisons = 5,
  verbose = TRUE)

#gráfico de pontos de pontuações para cada especificação de pré-processamento.
preText_score_plot(preText_results) 
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
text_df <- data_frame(line = 1:12134, text = text)
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

#remover casos em que qualquer um é um stop-word.
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

##### Correlação de palavras ####################### Atualizar version do R
#Quais pares de palavras aparecem
(ps_words <- tibble(chapter = seq_along(all.sam),
                    text = all.sam) %>%
    unnest_tokens(word, text) %>%
    filter(!word %in% stop_words$word))

#Podemos widyr p/ aproveitar o pacote para contar pares comuns de palavras que coexistem dentro do mesmo conjunto
install.packages("widyr")
library(widyr)


(word_pairs <- ps_words %>%
    pairwise_count(word, chapter, sort = TRUE))


#A saída fornece os pares de palavras como duas variáveis ( item1e item2). Isso nos permite realizar 
#atividades normais de mineração de texto, como procurar as palavras que mais freqüentemente seguem "Fogao"

word_pairs %>% 
  filter(item1 == "Fogao")

# Uma medida comum para essa correlação binária é o coeficiente phi .

#A pairwise_cor() função em widyr nos permite encontrar a correlação entre palavras com base na frequência 
#com que elas aparecem na mesma seção. Sua sintaxe é semelhante à pairwise_count().

(word_cor <- ps_words %>%
    group_by(word) %>%
    filter(n() >= 20) %>%
    pairwise_cor(word, chapter) %>%
    filter(!is.na(correlation)))

#Semelhante ao anterior, podemos agora avaliar a correlação de palavras de interesse. 
#Por exemplo, quais são as palavras correlacionadas mais altas que aparecem com "Fogao"?
word_cor %>%
  filter(item1 == "Fogao") %>%
  arrange(desc(correlation))

#Semelhante à forma como usamos o ggraph para visualizar bigrams, podemos usá-lo para visualizar as 
#correlações dentro dos clusters de palavras. Aqui, olhamos redes de palavras onde a correlação é bastante alta (> 0,65). 
#Podemos ver vários clusters aparecerem.

#Este tipo de gráfico fornece um ótimo ponto de partida para encontrar relacionamentos de conteúdo dentro do texto.
set.seed(123)

ps_words %>%
  group_by(word) %>%
  filter(n() >= 20) %>%
  pairwise_cor(word, chapter) %>%
  filter(!is.na(correlation),
         correlation > .65) %>%
  graph_from_data_frame() %>%
  ggraph(layout = "fr") +
  geom_edge_link(aes(edge_alpha = correlation), show.legend = FALSE) +
  geom_node_point(color = "lightblue", size = 5) +
  geom_node_text(aes(label = name), repel = TRUE) +
  theme_void()






