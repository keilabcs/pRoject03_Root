####Instalando pacotes pRoject03

install.packages("caret")
install.packages("lattice")
install.packages("ggplot2")
install.packages("Rtsne")
install.packages("ngram")
install.packages("RWeka")
install.packages("rJava",type='source')
install.packages("RWekajars")
install.packages("slam")
install.packages("tm")
install.packages("NLP")

library(NLP)
library(tm)
library(slam)
library(RWekajars)
library(rJava)
library(RWeka)
library(rJava)
library(ngram)
library(lattice)
library(ggplot2)
library(Rtsne)
library(caret)

##### LENDO BIBLIOTECAS

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

################################### Open

libs <- c("tm", "plyr", "class", "RWeka", "wordcloud")
lapply(libs, require, character.only = TRUE)

#Ler dados
X1 <- UFAL_2016_TODOS_SEM_84219999_REVISADO_CLASSIFICADA
X1

#Definir ######## NCM GERAL ############
B <- X1$DESCR_PROD 
B


all.sam <- c(B)
all.sam <- iconv(all.sam, 'UTF-8', 'ASCII') #Remove caracteres especiais

all.sam

#Crie uma função que limpe um corpus
clean_corpus <- function(corpus){
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, stripWhitespace)
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus, removeWords, stopwords("pt"))
  corpus <- tm_map(corpus, stemDocument)
  return(corpus)
}

#Convert para corpus e TDM
cor <- VCorpus(VectorSource(all.sam))
cor.cl <- clean_corpus(cor)
tdm <- TermDocumentMatrix(cor.cl)

tdm

#<<TermDocumentMatrix (terms: 8511, documents: 65120)>>
#Non-/sparse entries: 275212/553961108
#Sparsity           : 100%
#Maximal term length: 35
#Weighting          : term frequency (tf)

#convert to matrix
tdm.m <- as.matrix(tdm)

tdm.m

#faixa frequência de termos únicos
term.freq <- rowSums(tdm.m)
term.freq <- sort(term.freq,  decreasing = TRUE)
word.freqs <- data.frame(term = names(term.freq), num = term.freq)
word.freqs


#Explore a frequência dos termos com wordclouds e gráficos de barras
wordcloud(word.freqs$term, word.freqs$num, max.words = 100, colors="red")

barplot(word.freqs$num[1:25], names.arg = word.freqs$term[1:25], las = 2)

# Grafico de Barra ggplot p/ ferquencia 
ggplot(head(word.freqs, 30), aes(reorder(term,num), num)) +
  geom_bar(stat = "identity") + coord_flip() +
  xlab("Terms") + ylab("Frequency") +
  ggtitle("Most Frequent Terms")


#O objetivo final deste projeto é prever uma palavra ou palavras que virão em seguida para um determinado usuário.
#Portanto, pode ser útil aprender quais frases são frequentemente usadas ou quais os termos geralmente aparecem em 
#ordem juntos. Para isso, precisamos olhar N-gramas e com N-gramas você quer fazer um tokenizador. 
#O tokenizador irá definir quantas palavras usar em uma frase.

#Para criar um N grama, precisamos criar uma nova matriz de termo usando várias palavras.

# Criar tokenizador bi e tri gram
bitokenizer <- function(x)
  NGramTokenizer(x, Weka_control(min = 2, max =2))

tritokenizer <- function(x)
  NGramTokenizer(x, Weka_control(min = 3, max =3))

#Crie uma matriz de frases de duas palavras
tdm


bigram.tdm <- TermDocumentMatrix(cor.cl, control = list(tokenize = bitokenizer))
bi.words <- as.matrix(bigram.tdm)
bi.freq <- rowSums(bi.words)
bi.freq <- sort(bi.freq, decreasing = TRUE)
bi.word.freq <- data.frame(term = names(bi.freq), num = bi.freq)

bi.word.freq

#Crie uma matriz de frases de três palavras
trigram.tdm <- TermDocumentMatrix(cor.cl, control = list(tokenize = tritokenizer))
tri.words <- as.matrix(trigram.tdm)
tri.freq <- rowSums(tri.words)
tri.freq <- sort(tri.freq, decreasing = TRUE)
tri.word.freq <- data.frame(term = names(tri.freq), num = tri.freq)

tri.word.freq

#Examine a nova matriz e wordcloud
bigram.tdm

#<<TermDocumentMatrix (terms: 17984, documents: 65120)>>
#Non-/sparse entries: 255596/1170862484
#Sparsity           : 100%
#Maximal term length: 45
#Weighting          : term frequency (tf)

trigram.tdm

## WordCloud p/ bi-gram
wordcloud(bi.word.freq$term, bi.word.freq$num, max.words = 100, colors="red")

## Grafico de Barra p/ bi-grams
barplot(bi.word.freq$num[1:25], names.arg = bi.word.freq$term[1:25], las = 3)

## Grafico de Barra p/ bi-gram ggplot verical
ggplot(head(bi.word.freq, 50), aes(reorder(term,num), num)) +
  geom_bar(stat = "identity") + coord_flip() +
  xlab("Bigrams") + ylab("Frequency") +
  ggtitle("Most Frequent Bigrams")


## WordCloud p/ Tri-gram
wordcloud(tri.word.freq$term, tri.word.freq$num,max.words = 100, colors="red")

## Grafico de Barra p/ Tri-grams
barplot(tri.word.freq$num[1:25], names.arg = tri.word.freq$term[1:25], las = 3)

## Grafico de Barra p/ Tri-gram ggplot verical
ggplot(head(tri.word.freq, 50), aes(reorder(term,num), num)) +
  geom_bar(stat = "identity") + coord_flip() +
  xlab("Trigrams") + ylab("Frequency") +
  ggtitle("Most Frequent Trigrams")



############ Convertendo os dados de bi-grams p/ usa no ggplot Grafo ###############################

text <- B
text 

#Transformá-lo em um conjunto de dados de texto arrumado
text_df <- data_frame(line = 1:65120, text = text)
text_df

#Converter isso para que ele tenha um-token-por-documento-por-linha
#O processo de transformação do texto em bruto para unidades úteis para análise de texto é chamado de tokenização 
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


################ Visualizando uma rede de bigrams com ggraph ################################

bigram_counts

bigram_graph <- bigram_counts %>%
  filter(n > 50) %>%
  graph_from_data_frame()

bigram_graph

set.seed(2017)
ggraph(bigram_graph, layout = "fr") +
  geom_edge_link() +
  geom_node_point() +
  geom_node_text(aes(label = name), vjust = 1, hjust = 1)

#Agora, para visualizar nossa rede, aproveitaremos o ggraphpacote que converte um objeto igraf 
#para um gráfico de ggplot.

set.seed(123)

a <- grid::arrow(type = "closed", length = unit(.15, "inches"))

ggraph(bigram_graph, layout = "fr") +
  geom_edge_link() +
  geom_node_point(color = "lightblue", size = 5) +
  geom_node_text(aes(label = name), vjust = 1, hjust = 1) +
  theme_void()


############ Convertendo os dados de tri-grams p/ usa no ggplot Grafo ###############################

text <- B
text 

#Transformá-lo em um conjunto de dados de texto arrumado
text_df <- data_frame(line = 1:65120, text = text)
text_df

#Converter isso para que ele tenha um-token-por-documento-por-linha
text_df %>%
  unnest_tokens(word, text)

text_df 

#Tokenizing por n-grama
austen_trigrams <- text_df %>%
  unnest_tokens(trigram, text, token = "ngrams", n = 3)

austen_trigrams

#Análise de n-grama
austen_trigrams %>%
  count(trigram, sort = TRUE)

#remover casos em que qualquer um é um stop-word.
trigrams_separated <- austen_trigrams %>%
  separate(trigram, c("word1", "word2", "word3"), sep = " ")

trigrams_filtered <- trigrams_separated %>%
  filter(!word1 %in% stop_words$word) %>%
  filter(!word2 %in% stop_words$word)%>%
  filter(!word3 %in% stop_words$word)

# new bigram counts:
trigram_counts <- trigrams_filtered %>% 
  count(word1, word2, word3, sort = TRUE)

trigram_counts


################ Visualizando uma rede de trigram com ggraph ################################

trigram_counts

trigram_graph <- trigram_counts %>%
  filter(n > 50) %>%
  graph_from_data_frame()

trigram_graph

set.seed(2017)
ggraph(trigram_graph, layout = "fr") +
  geom_edge_link() +
  geom_node_point() +
  geom_node_text(aes(label = name), vjust = 1, hjust = 1)

#Agora, para visualizar nossa rede, aproveitaremos o ggraphpacote que converte um objeto igraf 
#para um gráfico de ggplot.

set.seed(123)

a <- grid::arrow(type = "closed", length = unit(.15, "inches"))

ggraph(trigram_graph, layout = "fr") +
  geom_edge_link() +
  geom_node_point(color = "lightblue", size = 5) +
  geom_node_text(aes(label = name), vjust = 1, hjust = 1) +
  theme_void()


#### Modelo n-grama e previsão ####################### Construindo um Modelo de Previsão de texto com base no n-grams

#Com base nessa análise exploratória, agora esboço um algoritmo básico para a predição de texto usando tabelas de n gramas.

#1, 2, 3 e 4 tabelas de n grama são armazenadas como arquivos de texto.
#Somente os n-gramas com uma qualidade superior ou igual a 2 são mantidos no modelo.
#As tabelas de n grama são carregadas a partir dos arquivos de texto.
#Para uma seqüência de texto que é inserida no preditor, o algoritmo de previsão realiza uma pesquisa em cada tabela de n grama, começando com a tabela de 4 gramas.
#A partir do texto imput, os últimos três termos são obtidos e pesquisados na tabela de 4 gramas. Se uma ou mais combinações forem encontradas, o algoritmo exibirá as melhores previsões para a próxima palavra, dado esses três termos.
#Se nenhuma correspondência for encontrada na tabela de 4 gramas, a busca continua na tabela de 3 gramas usando as duas últimas palavras da entrada. E assim por diante. Se nenhuma correspondência for encontrada, a previsão é, então, o mais comum de um grama (termos únicos).
#Por exemplo, uma previsão para "e um caso de" seria:
  
#ANLP é um pacote que fornece todas as funcionalidades para construir modelo de previsão de texto.
install.packages("ANLP")
library(ANLP)

print(length(all.sam))
all.sam

#Precisamos amostrar 10% dos dados.
train.data <- sampleTextData(all.sam,0.5)
print(length(train.data))

head(train.data)

#CleanTextData
train.data.cleaned <- cleanTextData(train.data)
train.data.cleaned[[1]]$content[1:1]

#criar modelos N-gram usando nosso corpus de dados limpos.
#Vamos construir modelos de 1,2,3 gramas e gerar matriz de freqüência de termo para todos os dados.
unigramModel <- generateTDM(train.data.cleaned,1)
head(unigramModel)

bigramModel <- generateTDM(train.data.cleaned,2)
head(bigramModel)

trigramModel <- generateTDM(train.data.cleaned,3)
head(trigramModel)


#Lembre-se de unir os modelos N-gram em ordem decrescente. (3,2,1 modelos Ngram)
nGramModelsList <- list(bi.word.freq,tri.word.freq,word.freqs)
nGramModelsList

# Vamos prever algumas cordas:
testString <- "fogao gas 4b esmaltec"
predict_Backoff(testString,nGramModelsList)
