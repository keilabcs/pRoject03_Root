### Projeto3

##### CARREGANDO PACOTES ##### LENDO BIBLIOTECAS

install.packages("ggplot2")
library(ggplot2)

install.packages("ggthemes") #### Camadas
library(ggthemes)

install.packages("extrafont")
library(extrafont)

install.packages("tidyr") ##### Mudar Cor 
library(tidyr)

install.packages("wesanderson")
library(wesanderson)

install.packages("Rcmdr") # Problema na instalacao
library(Rcmdr)

install.packages("ISwR")
library(ISwR)

install.packages("tools")
library(tools)

install.packages("gridExtra")
library(gridExtra)

install.packages("grid")
library(grid)

install.packages("NLP") # ngrams
library(NLP)

#Você precisa fechar todas as sessões R também do R Studio.
#Abra o terminal e digite xcode-select --install
#Tipo curl -O http://r.research.att.com/libs/gfortran-4.8.2-darwin13.tar.bz2
#Tipo sudo tar fvxz gfortran-4.8.2-darwin13.tar.bz2 -C /
# Você pode instalar 'slam' agora.
install.packages("slam")
library("slam")

install.packages("tm")
library(tm)

install.packages("SnowballC")
library(SnowballC)

install.packages("wordcloud")
library(wordcloud)

install.packages("wordcloud2")
library(wordcloud2)

install.packages("igraph") #Grafo
library(igraph)

install.packages("readr") #tokenize
library(readr) #tokenize

install.packages("utils") #View
library(utils) #View

install.packages("factoextra") # Install factoextra
library(factoextra)

install.packages("cluster") # Install cluster package
library(cluster)

install.packages("preText") #PreText 2
library(preText)

#O pre-processamento de textos e uma das principais etapas da mineracao de textos, e tambem uma das mais
#custosas. Essa etapa visa transformar texto nao estruturado em um formato estruturado, como uma tabela 
#atributo-valor. O PreTexT e uma ferramenta computacional que realiza esse tipo de pre-processamento utilizando 
#funcionalidades como n-grama, stemming, stoplists, cortes por frequencia, taxonomias, normalizacoes,
#graficos, medidas tf , tf-idf , tf-linear , boolean, entre outras.

##### LENDO BIBLIOTECAS

library(plyr)
library(quanteda)


#########################################################################################
#Tecnicas para a redução da quantidade de atributos (dimensão)

#case folding : Converte os caracteres p/ minusculo.
#stopwords : Preposicoes, pronomes, artigos, adverbios.
#stemming : Reduz ao Radical.
#dicionários (thesaurus) : Mapeia sinônimos, acrônimos e ortografias.
#redução de atributos por medidas de relevância: TF-IDF, Lei de Zipf.
###########################################################################################

########## Lendo a Base

X1 <- UFAL_2016_TODOS_SEM_84219999_REVISADO_CLASSIFICADA
X1

###################### Nuvem de palavras

#######################
B <- X1$DESCR_PROD [X1$NCM_PROD == 73211100] #DELIMITAR POR NCM
B

###################### Remocao dos acentos 

rm_accent <- function(B, pattern="all") {
  # Rotinas e funções úteis V 1.0
  # rm.accent - REMOVE ACENTOS DE PALAVRAS
  # Função que tira todos os acentos e pontuações de um vetor de strings.
  # Parâmetros:
  # str - vetor de strings que terão seus acentos retirados.
  # patterns - vetor de strings com um ou mais elementos indicando quais acentos deverão ser retirados.
  #            Para indicar quais acentos deverão ser retirados, um vetor com os símbolos deverão ser passados.
  #            Exemplo: pattern = c("´", "^") retirará os acentos agudos e circunflexos apenas.
  #            Outras palavras aceitas: "all" (retira todos os acentos, que são "´", "`", "^", "~", "¨", "ç")
  if(!is.character(B))
    str <- as.character(B)
  
  pattern <- unique(pattern)
  
  if(any(pattern=="Ç"))
    pattern[pattern=="Ç"] <- "ç"
  
  symbols <- c(
    acute = "áéíóúÁÉÍÓÚýÝ",
    grave = "àèìòùÀÈÌÒÙ",
    circunflex = "âêîôûÂÊÎÔÛ",
    tilde = "ãõÃÕñÑ",
    umlaut = "äëïöüÄËÏÖÜÿ",
    cedil = "çÇ"
  )
  
  nudeSymbols <- c(
    acute = "aeiouAEIOUyY",
    grave = "aeiouAEIOU",
    circunflex = "aeiouAEIOU",
    tilde = "aoAOnN",
    umlaut = "aeiouAEIOUy",
    cedil = "cC"
  )
  
  accentTypes <- c("´","`","^","~","¨","ç")
  
  if(any(c("all","al","a","todos","t","to","tod","todo")%in%pattern)) # opcao retirar todos
    return(chartr(paste(symbols, collapse=""), paste(nudeSymbols, collapse=""), B))
  
  for(i in which(accentTypes%in%pattern))
    B <- chartr(symbols[i],nudeSymbols[i], B)
  
  return(B)
}

B

write.table(B, "/users/keila/B.txt", sep="\t") #Guarda a base txt

###################### Técnicas para a redução da quantidade de atributos #############

#Mineracao
aux <- B
auxCorpus <- Corpus(VectorSource(aux)) #criar um corpus

auxCorpus <- tm_map(auxCorpus, PlainTextDocument) #converter o corpus em um documento de texto simples
auxCorpus = tm_map(auxCorpus, content_transformer(tolower)) #convert para minúsculas
auxCorpus <- tm_map(auxCorpus, removePunctuation) #remove punctuation
auxCorpus <- tm_map(auxCorpus, removeNumbers) # remove numbers
auxCorpus <- tm_map(auxCorpus, stemDocument) #verificar as palavras e seus sinônimos... (STEMMER) portugues
#diferentes formas da palavra em questão serão convertidos para a mesma forma

auxCorpus <- tm_map(auxCorpus, removeWords, stopwords('pt')) #Remove Conjuncoes... stopword list

auxCorpus <- Corpus(VectorSource(auxCorpus))
auxCorpus

matriz_terms <- DocumentTermMatrix(auxCorpus)
matriz_terms

#<<DocumentTermMatrix (documents: 3, terms: 565)>>
#  Non-/sparse entries: 568/1127
#Sparsity           : 66%
#Maximal term length: 16
#Weighting          : term frequency (tf)

wordcloud(auxCorpus,colors=c("blue","red"))


#Criar uma matriz de termo-documento frequencia
dtm <- TermDocumentMatrix(auxCorpus)
m <- as.matrix(dtm)
v <- sort(rowSums(m),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v)
head(d, 10)

#word  freq
#fogao       fogao 11097
#esmaltec esmaltec  3385
#branco     branco  2511
#atlas       atlas  2256
#inox         inox  2219
#gas           gas  1513
#bco           bco  1443
#bali         bali  1243
#biv           biv  1145
#monaco     monaco   960


### Grafo

### Matriz de distância baseadas em correlação

## Frequencia das palavras da descricao Geral #####################################################
texto <- X1$DESCR_PROD
lista_palavras <- strsplit(texto, "\\W+")
vetor_palavras <- unlist(lista_palavras)

frequencia_palavras <- table(vetor_palavras)
frequencia_ordenada_palavras <- sort(frequencia_palavras, decreasing=TRUE)

palavras <- paste(names(frequencia_ordenada_palavras), frequencia_ordenada_palavras, sep=";")

cat("Palavra;Frequencia", palavras, file="frequencias.csv", sep="\n") #Salvando a Freq Geral na Base


######################################################################################################

#################### Lei de Zipf
# Obs. falta desenvolver os cortes pela Lei de Zipf...

