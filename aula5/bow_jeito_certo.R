# ---
# _RODANDO MODELOS BOW DO JEITO CERTO COM MLR3!
# ---


# Pacotes
library(tidyverse)
library(mlr3verse)
library(quanteda)


# Vamos carregar os dados com:
discursos <- "https://github.com/FLS-6497/datasets/blob/main/aula5/discursos_presidenciais.csv?raw=true" %>%
  read_csv2() %>%
  mutate(id = row_number()) # Criamos um ID


# Funcao para pre-processar textos
cria_bow <- function(df, var){
  
  # 1) Criar um corpus
  corpus_disc <- corpus(df, text_field = var)
  
  # 2) Tokeniza o corpus
  tks_disc <- corpus_disc %>%
    tokens(remove_punct = TRUE, remove_numbers = TRUE) %>%
    tokens_tolower() %>%
    tokens_remove(min_nchar = 5, pattern = stopwords("pt"))
  
  # 3) Cria uma matriz bag-of-words
  bow <- dfm(tks_disc) %>%
    dfm_trim(min_docfreq = 5)
  
  # 4) Transforma a bow para um formato que o mlr3 entenda
  dados <- as.matrix(bow) %>%
    as_tibble() %>%
    janitor::clean_names()
  
  # 5) Adiciona o target e retorna
  dados$y <- df$presidente
  return(list(df = dados, bow = bow))
}


# Separa a amostra em treino e teste
treino <- discursos %>%
  sample_frac(0.7)
  
teste <- discursos %>%
  filter(!id %in% treino$id)


# Transforma a amostra de treino em BOW
treino_bow <- cria_bow(treino, "discurso")


# Aplica a estrutura da amostra de treino na de teste
teste_bow <- teste %>%
  corpus(text_field = "discurso") %>%
  tokens() %>%
  dfm() %>%
  dfm_match(featnames(treino_bow$bow)) %>%
  as.matrix() %>%
  as_tibble() %>%
  janitor::clean_names()
teste_bow$y <- as.factor(teste$presidente)


# Treina um modelo Naive Bayes
tsk <- as_task_classif(y ~ ., data = treino_bow$df)
learner <- lrn("classif.naive_bayes")
learner$train(tsk)

# Faz predicoes
pred <- learner$predict_newdata(teste_bow)
pred 

# Adiciona as predicoes na base de teste pra visualizarmos ela!
teste$predicao <- pred$response
View(teste)


# Confere predicoes com metricas de validacao
pred$confusion
pred$score(msrs(c("classif.precision", "classif.recall", "classif.fbeta")))


