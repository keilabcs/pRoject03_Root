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
install.packages("RISmed")
install.packages("stats") 
install.packages("base") 
install.packages("dplyr")

##### LENDO BIBLIOTECAS

library(RISmed)
library(NLP)
library(tm)
library(slam)
library(RWekajars)
library(rJava)
library(RWeka)
library(rJava)
library(ngram)
library(caret)
library(lattice)
library(Rtsne)
library(SnowballC)
library(wordcloud)
library(igraph)
library(cluster)
library(factoextra)
library(readr) #tokenize
library(utils) #View
library(preText)
library(quanteda)
library(tidytext)
library(janeaustenr)
library(tidyr)
library(ggraph)
library(dplyr)
library(ggplot2)
library(gridExtra)

libs <- c("tm", "plyr", "class", "RWeka", "wordcloud")
lapply(libs, require, character.only = TRUE)

getwd() #Verificar diretorio usado pelo r 

################################### Open


################################################# Inicio de Predicion NCM 73211100 
# Um algoritmo de predição de palavras.
#Nossos propósitos de tentar construir um preditor de próxima palavra.

#Defina as opções de knitr para usar ao longo
options(width=120)

#Ler dados
X1 <- UFAL_2016_TODOS_SEM_84219999_REVISADO_CLASSIFICADA
X1

#Definir NCM
B <- X1$DESCR_PROD[X1$NCM_PROD == 73211100] #DELIMITAR POR NCM
B

all.sam <- c(B)
all.sam <- iconv(all.sam, 'UTF-8', 'ASCII') #Remove caracteres especiais

all.sam


# Dê uma olhada rápida nos quadros de dados
head(B)
head(all.sam)

all.sam

#Tokenização de Cadeia
#Para o nosso modelo de predição de ngrama, consideramos palavras inteiras, 
#com números considerados como palavras, e vamos adicionar o token '## s ##' no início das frases 
#(ou fragmentos de sentença) e o Token '## es ##' no final das frases (ou fragmentos de sentença). 
#(Nosso modelo de previsão também incluirá um token '## unkn ##' para palavras desconhecidas).

#Como tal, precisamos dividir o texto em palavras e também adicionar os toques "## s ##" e "## es ##" nos 
#locais apropriados. A seguinte função leva uma string como entrada e retorna uma versão tokenizada dela, 
#e todas as letras minúsculas.

wordList <- function(str) {
  str = gsub("[.!?;]", " ##es## ##s## ", str)
  str = paste("##s##",str,"##es##")
  str = gsub("##s##[ ]+##es##", "", str)
  regex = "[^[:space:],:=<>/\\)\\(]+"
  regmatches(tolower(str), gregexpr(regex, tolower(str)))
}

#Um exemplo de uso da função wordList:
all.sam[4]
wordList(all.sam[4])

#Podemos então aplicar esta função a cada linha de cada uma das nossas amostras de dados:
wordsSeqs_NCM <- sapply(all.sam,wordList,USE.NAMES=FALSE)

#Uma rápida olhada nos resultados:
wordsSeqs_NCM[1:2]

#Análise exploratória
#Tokens individuais

#Nosso primeiro passo na análise será considerar ocorrências de tokens individuais, que chamaremos o caso N1.
wordsN1 <- c(unlist(wordsSeqs_NCM))
totalWords = length(wordsN1)
totalWords

#Podemos então usar o seguinte para calcular a frequência de cada palavra (token) nesta lista:
wordCountsN1 <- as.data.frame(table(wordsN1))
wordCountsN1$Freq <- wordCountsN1$Freq / totalWords

#Olhar rápido sobre os tokens mais frequentes:
wordCountsN1_desc <- wordCountsN1 %>% arrange(desc(Freq))
head(wordCountsN1_desc, n=10)

#Para a nossa aplicação, queremos considerar a redução do número de tokens em nossa coleção, 
#substituindo algumas palavras incomuns pelo token único ## unkn ##. Isso pode diminuir 
#significativamente os requisitos de memória para o aplicativo com apenas uma diminuição de precisão 
#relativamente pequena, e também permite o tratamento de casos em que os usuários escrevem palavras que 
#não estão nos corpos. O seguinte calcula o número de tokens necessários para capturar 50% ou 90% da contagem
#de token, e a figura mostra as distribuições em maior detalhe.
wordsReq50 = min(which( cumsum(wordCountsN1_desc$Freq) > 0.5 ))
wordsReq90 = min(which( cumsum(wordCountsN1_desc$Freq) > 0.9 ))
wordsReq50
wordsReq90

# Graficos
wordCountsN1_asc <- wordCountsN1 %>% arrange(Freq)
wordCountsN1_asc$wordsN1 <- factor(wordCountsN1_asc$wordsN1,
                                   levels = wordCountsN1_asc$wordsN1, 
                                   ordered = TRUE)

y1 <- ggplot(wordCountsN1_asc, 
             aes(x=wordsN1,y=Freq)) +
  geom_bar(stat="identity") + coord_flip() + 
  xlab("Todos os tokens ordenados pela prevalência") + ylab("Frequência") +
  theme(axis.ticks.y = element_blank(), axis.text.y = element_blank()) +
  theme(text = element_text(size=9))

y1

y2 <- ggplot(tail(wordCountsN1_asc,n=30), 
             aes(x=wordsN1,y=Freq)) +
  geom_bar(stat="identity") + coord_flip() + 
  xlab("30 tokens mais comumente usados") + ylab("Frequency") +
  theme(text = element_text(size=9))

y2

y3 <- ggplot(tail(wordCountsN1_asc,n=30), 
             aes(x=1:30,y=(sum(Freq)-cumsum(Freq)))) +
  geom_line() + coord_flip() + 
  xlab("30 tokens mais comumente usados") + ylab("Soma Cummulativa de Frequências") +
  scale_x_discrete(breaks=1:30, labels=tail(wordCountsN1_asc$wordsN1,n=30)) +
  theme(text = element_text(size=9))

y3

y4 <- ggplot(wordCountsN1_asc, 
             aes(x=1:length(wordsN1),y=(1-cumsum(Freq)))) +
  geom_line() + coord_flip() + 
  xlab("Todos os tokens ordenados pela prevalência") + ylab("Soma Cummulativa de Frequências") +
  theme(axis.ticks.y = element_blank(), axis.text.y = element_blank()) +
  theme(text = element_text(size=9)) + 
  geom_vline(aes(xintercept=length(wordCountsN1_asc$wordsN1)-wordsReq50,
                 colour='blue')) + 
  geom_vline(aes(xintercept=length(wordCountsN1_asc$wordsN1)-wordsReq90,
                 colour='red')) +
  geom_text(aes(length(wordCountsN1_asc$wordsN1)-wordsReq50, 1,
                label = paste("50%: ", wordsReq50, " tokens"), 
                hjust = 1, vjust = -1, size=9, colour='blue')) +
  geom_text(aes(length(wordCountsN1_asc$wordsN1)-wordsReq90, 1,
                label = paste("90%: ", wordsReq90, " tokens"), 
                hjust = 1, vjust = -1, size=9, colour='red')) +
  theme(legend.position = 'none')

y4

grid.arrange(y1, y2, y3, y4, ncol=4)

#N-gramas - Grupos de Tokens

#A seguinte função leva uma lista de palavras (por exemplo, nossa lista de palavrasN1) e produz uma lista 
#de pares (2 gramas), triplos (3 gramas) ou, mais geralmente, n-gramas. Esses n-gramas serão usados em nosso 
#aplicativo de previsão, e também podem ser usados em várias outras aplicações.
makeNlist <- function(x,n) {
  l <- list()
  l[[1]] <- x
  for (i in 2:n) {
    l[[i]] <- l[[i-1]]
    l[[i]] <- l[[i]][-1]
    l[[i]][length(l[[i]])+1] <- "##es##"
  }
  df <- data.frame(l)
  colnames(df) <- 1:n
  do.call("paste", c(df[1:n], sep=" "))
}

#Com essa lista, podemos repetir a análise N1 anterior para qualquer lista de N grama usando a 
#seguinte função:
analyseNgramList <- function(wordsNgram) {
  
  totalWords = length(wordsNgram)
  print(paste("total N-grams = ", totalWords, sep=""))
  
  wordCountsNgram <- as.data.frame(table(wordsNgram))
  wordCountsNgram$Freq <- wordCountsNgram$Freq / totalWords
  distinctWords = length(wordCountsNgram$Freq)
  print(paste("distinct N-grams = ", distinctWords, sep=""))
  
  wordCountsNgram_desc <- wordCountsNgram %>% arrange(desc(Freq))
  
  wordsNgramReq50 = min(which( cumsum(wordCountsNgram_desc$Freq) > 0.5 ))
  wordsNgramReq90 = min(which( cumsum(wordCountsNgram_desc$Freq) > 0.9 ))
  print(paste("num N-grams required for 50% coverage= ", wordsNgramReq50, sep=""))
  print(paste("num N-grams required for 90% coverage= ", wordsNgramReq90, sep=""))
  
  wordCountsNgram
} 


graphNgramList <- function(wordsNgram,wordCountsNgram) {
  
  wordCountsNgram_asc <- wordCountsNgram %>% arrange(Freq)
  wordCountsNgram_asc$wordsNgram <- factor(wordCountsNgram_asc$wordsNgram,
                                           levels = wordCountsNgram_asc$wordsNgram, 
                                           ordered = TRUE)
  
  y1 <- ggplot(wordCountsNgram_asc, 
               aes(x=wordsNgram,y=Freq)) +
    geom_bar(stat="identity") + coord_flip() + 
    xlab("Todos os N-gramas ordenados pela prevalência") + ylab("Frequency") +
    theme(axis.ticks.y = element_blank(), axis.text.y = element_blank()) +
    theme(text = element_text(size=9))
  
  y2 <- ggplot(tail(wordCountsNgram_asc,n=30), 
               aes(x=wordsNgram,y=Freq)) +
    geom_bar(stat="identity") + coord_flip() + 
    xlab("30 N-gramas mais comumente usados") + ylab("Frequency") +
    theme(text = element_text(size=9))
  
  y3 <- ggplot(tail(wordCountsNgram_asc,n=30), 
               aes(x=1:30,y=(sum(Freq)-cumsum(Freq)))) +
    geom_line() + coord_flip() + 
    xlab("30 N-gramas mais comumente usados") + ylab("Soma Cummulativa de Frequências") +
    scale_x_discrete(breaks=1:30, labels=tail(wordCountsNgram_asc$wordsNgram,n=30)) +
    theme(text = element_text(size=9))
  
  y4 <- ggplot(wordCountsNgram_asc, 
               aes(x=1:length(wordsNgram),y=(1-cumsum(Freq)))) +
    geom_line() + coord_flip() + 
    xlab("Todos os N-gramas ordenados pela prevalência") + ylab("Soma Cummulativa de Frequências") +
    theme(axis.ticks.y = element_blank(), axis.text.y = element_blank()) +
    theme(text = element_text(size=9)) 
  
  list(y1,y2,y3,y4)
}

#2 gramas

#Quais são as frequências de 2 gramas no conjunto de dados?
wordsN2 <- makeNlist(wordsN1,2)
wordsN2[1:10]

wordCountsN2 <- analyseNgramList(wordsN2)

y <- graphNgramList(wordsN2,wordCountsN2)
grid.arrange(y[[1]], y[[2]], y[[3]], y[[4]], ncol=4)

#3 gramas

#Quais são as frequências de 3 gramas no conjunto de dados?
wordsN3 <- makeNlist(wordsN1,3)
wordsN3[1:10]

wordCountsN3 <- analyseNgramList(wordsN3)
y <- graphNgramList(wordsN3,wordCountsN3)
grid.arrange(y[[1]], y[[2]], y[[3]], y[[4]], ncol=4)


#Plano de desenvolvimento para criar um algoritmo de previsão e uma aplicação brilhante
#Algoritmo básico de predição de n grama, assumindo que o dado (n-1) -gram está nos corpos:

predict <- function(given2gram,wordCounts,n) {
  found <- grepl(paste("^",given2gram," ",sep=""),wordCounts$wordsNgram)
  df <- wordCounts[found,]
  df$Freq = df$Freq / sum(df$Freq)
  df$wordsNgram = gsub(paste("^",given2gram," ",sep=""),"",df$wordsNgram)
  head(df[ order(-df[,2]), ], n=n)
}

w <- predict("fogao",wordCountsN3,5)
w

w <- predict("fogao 4b",wordCountsN3,5)
w

w <- predict("fg",wordCountsN3,5)
w

w <- predict("fg gas",wordCountsN3,5)
w

w <- predict("kit",wordCountsN3,5)
w

w <- predict("kit luxo",wordCountsN3,5)
w

w <- predict("kit classic",wordCountsN3,5)
w

#Criando um aplicativo Shiny p o NCM 73211100
install.packages("shiny")
library(shiny)
install.packages('rsconnect')
library(rsconnect)
install.packages(c("ggplot2", "reshape2", "dplyr"))

rsconnect::setAccountInfo(name='keilalaccan',
                          token='69F6B27A8B0C31520CB098CB1827675A',
                          secret='<SECRET>')

library(shiny)

ui <- fluidPage(
  textInput("text", label = h2("NCM Next Word Predictor"), value = " "),
  submitButton(text = "Predict next word..."),
  hr(),
  fluidRow((verbatimTextOutput("value")))
)


server <- function(input, output) {
  output$value <- renderPrint({
    input$text
  })
}


shinyApp(ui = ui, server = server)




