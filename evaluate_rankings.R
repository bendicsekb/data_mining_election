library(ggplot2)
library(mice)
library(tidyverse)

# Working directory
setwd("C:/Users/20200059/Documents/Github/data_mining_election/RealWorld/results/")

data <- read.csv("../Demographic_and_election_dataset_ranked.csv", check.names=FALSE)

head(data)
md.pattern(data) # no missing values

colnames <- names(data)
colnames[!sapply(data, is.numeric)]
colnames

mutdata <- data %>%
  #slice(c(1:106),c(108:351)) %>%
  mutate(`Newly constructed houses (%)` = as.numeric(`Newly constructed houses (%)`)) %>%
  mutate(`Companies by type - agriculture, forestry and fishery (%)` = as.numeric(as.character(`Companies by type - agriculture, forestry and fishery (%)`))) %>%
  rename(`Companies by type - trade, catering industry (%)` = `Companies by type - trade and catering industry (%)`) %>%
  rename(`Companies by type - agriculture, forestry, fishery (%)` = `Companies by type - agriculture, forestry and fishery (%)`) %>%
  rename(`Companies by type - industry, engery (%)` = `Companies by type - industry and engery (%)`) %>%
  rename(`Households wh children (%)` = `Households without children (%)`) %>%
  rename(`Companies by type - financial services, real-estate (%)` = `Companies by type - financial services and real-estate (%)`) %>%
  rename(`Companies by type - transport, information, comunication (%)` = `Companies by type - transport, information and comunication (%)`) %>%
  rename(`Companies by type - culture, recreation, other (%)` = `Companies by type - culture, recreation and other (%)`) %>%
  rename(`Households w children (%)` = `Households with children (%)`)

votinglist <- tail(colnames,37)

# overall ranking in the dataset
pid <- unname(rank(colSums(mutdata[,votinglist])))

# make one dataset and one plot
files <- c("entropy.txt", "entropy_compl.txt", 
           "sqrtnN.txt", "sqrtnN_compl.txt",
           "none.txt", "none_compl.txt",
           "labelwise.txt", "norm.txt", "pairwise.txt")

alldata <- as.data.frame(matrix(data=NA,nrow=0,ncol=38))
for(i in 1:length(files)){
  file <- files[i]
  print(file)
  q <- 50
  results <- readLines(file, n=(q+1))[2:(q+1)]
  if(sub(".*\\_", "", sub("\\..*", "", file)) == 'compl'){
    figuredata <- as.data.frame(matrix(data=NA,nrow=q,ncol=39)) %>%
      mutate(sg = c(1:q))
    for(qi in 0:(q-1)){
      #print(qi)
      result <- results[q-qi]
      #print(result)
      desc <- sub("\\ with.*", "", result)
      #print(desc)
      lits <- strsplit(desc,split=" and ")
      lits[[1]][1] <- sub(".*: ", "", lits[[1]][1])
      subset <- mutdata[0,]
      for(li in 1:length(lits[[1]])){
        #print(li)
        lit <- lits[[1]][li]
        name <- sub("=.*", "", lit)
        name <- substr(name,1,nchar(name)-2)
        value <- as.numeric(sub(".*= ", "", lit))
        sign <- tail(strsplit(sub("=.*", "", lit), split="")[[1]],n=1)
        #print(name)
        #print(value)
        #print(sign)
        if(sign == "<"){
          idx <- mutdata[,name] <= value
          sel <- mutdata[!idx,]
        } else if(sign == ">"){
          idx <- mutdata[,name] >= value
          sel <- mutdata[!idx,]
        }
        subset <- rbind(subset,sel)
      }
      pi <- unname(rank(colSums(subset[,votinglist])))
      pidif <- pi - pid
      figuredata[qi+1,1:37] <- pidif
      figuredata[qi+1,38] <- nrow(subset)
    }
    figuredata[,39] <- rep(paste0(sub("\\..*", "", file)," ",""),q)
    alldata <- rbind(alldata, figuredata)
  }
  # do this for all results
  figuredata <- as.data.frame(matrix(data=NA,nrow=q,ncol=39)) %>%
    mutate(sg = c(1:q))
  for(qi in 0:(q-1)){
    #print(qi)
    result <- results[q-qi]
    #print(result)
    desc <- sub("\\ with.*", "", result)
    #print(desc)
    lits <- strsplit(desc,split=" and ")
    lits[[1]][1] <- sub(".*: ", "", lits[[1]][1])
    subset <- mutdata
    for(li in 1:length(lits[[1]])){
      #print(li)
      lit <- lits[[1]][li]
      name <- sub("=.*", "", lit)
      name <- substr(name,1,nchar(name)-2)
      value <- as.numeric(sub(".*= ", "", lit))
      sign <- tail(strsplit(sub("=.*", "", lit), split="")[[1]],n=1)
      #print(name)
      #print(value)
      #print(sign)
      if(sign == "<"){
        subset <- subset[subset[,name] <= value,]
      } else if(sign == ">"){
        subset <- subset[subset[,name] >= value,]
      }
    }
    pi <- unname(rank(colSums(subset[,votinglist])))
    pidif <- pi - pid
    figuredata[qi+1,1:37] <- pidif
    figuredata[qi+1,38] <- nrow(subset)
  }
  figuredata[,39] <- rep(sub("\\..*", "", file),q)
  alldata <- rbind(alldata, figuredata)
}

nrsg <- 20
plotdata <- alldata %>%
  filter(sg < nrsg+1) %>%
  gather("lambda", "rank", -c(sg, V38, V39)) %>%
  rename(size = V38, correction=V39) %>%
  mutate(rank = as.numeric(as.character(rank))) %>%
  mutate(lambda = ordered(lambda,
                          levels = sapply(c(1:37), function(x) paste0("V", as.character(x), "")))) %>%
  #mutate(method = sub("\\..*", "", method)) %>%
  mutate(comparison = sub(".*\\_", "", correction)) %>%
  mutate(correction = sub("\\_.*", "", correction)) %>%
  mutate(comparison = recode(comparison, "entropy" = "average", "sqrtnN" = "average", "none" = "average",
                             "labelwise" = "average", "pairwise" = "average", "norm" = "average")) %>%
  mutate(correction = ordered(correction,
                             levels = c("none", "entropy", "sqrtnN", "norm", "labelwise", "pairwise"))) %>%
  mutate(comparison = ordered(comparison,
                              levels = c("compl", "compl ", "average"))) %>%
  mutate(rank = replace(rank, rank > 10, 10))

# overall plot
plot <- plotdata %>%
  filter(correction %in% c("none", "entropy", "sqrtnN")) %>%
  #filter(correction %in% c("labelwise", "pairwise", "norm")) %>%
  ggplot(aes(y=sg, x=lambda, fill=rank)) + 
  geom_tile(color="white") + 
  facet_grid(comparison ~ correction, 
             labeller = label_both) +
             #labeller = labeller(correction = 
            #            c("norm" = "type: norm",
            #              "labelwise" = "type: labelwise",
            #              "pairwise" = "type: pairwise"),
            #            comparison = 
            #              c("average" = "comparison: average"))) + 
  scale_colour_gradient2(low = "#a63603", mid = "white",
                         high = "#1f78b4", midpoint = 0,
                         aesthetics = "fill",
                         limits=c(-10, 10)) + 
  scale_x_discrete(labels = as.character(c(1:37))) + 
  scale_y_continuous(expand = c(0, 0)) +
  guides(fill = guide_colorbar(barwidth = 3, barheight = 0.5)) + 
  xlab('Labels') + 
  ylab('Subgroup') +
  #ylab("Subgroup                                        Complement                                        Subgroup") +
  theme_bw() + 
  theme(plot.title = element_text(vjust=-4), 
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
    panel.grid.major.y = element_blank(),
    panel.grid.minor.y = element_blank(),
    panel.border = element_blank(),
    axis.line = element_line(colour = "black"),
    axis.text.x = element_text(size=4),
    axis.title = element_text(size=10),
    legend.title = element_blank(),#element_text(size = 6), 
    legend.text  = element_text(size = 4),
    legend.position="bottom",
    legend.justification="right",
    legend.margin=margin(-20, 0, 0, 0))
plot
name <- paste('ourqms.pdf', sep = "", collapse = NULL)
ggsave(name, width = 20, height = 18, units = "cm")
name <- paste('otherqms.pdf', sep = "", collapse = NULL)
ggsave(name, width = 20, height = 7, units = "cm")

alldata %>% filter(V39 == 'none') %>% filter(sg == 1)

# make separate plots
methods <- c("entropy", "none", "sqrtnN", "labelwise", "pairwise", "norm")
types <- c("compl", "compl ", "average")
for(i in 1:length(methods)){
  for(j in 1:length(types)){
    selmethod <- methods[i]
    if(selmethod %in% c("entropy", "none", "sqrtnN")){
      seltype <- types[j]
      seldata <- plotdata %>%
        filter(correction == selmethod) %>%
        filter(comparison == seltype)
      if(seltype == "compl "){
        seltype <- "compl.opp"
      }
    } else {
      seldata <- plotdata %>%
        filter(correction == selmethod)
    }
    plot <- seldata %>%
      ggplot(aes(y=sg, x=lambda, fill=rank)) + 
      geom_tile(color="white") + 
      scale_colour_gradient2(low = "red", mid = "white",
                             high = "blue", midpoint = 0,
                             aesthetics = "fill") + 
      scale_x_discrete(labels = as.character(c(1:37))) + 
      scale_y_continuous(expand = c(0, 0)) +
      guides(fill = guide_colorbar(barwidth = 0.4)) + 
      guides(fill = guide_legend(title = "Change")) + 
      labs(title = paste0("Change in rank with ", selmethod, " correction", ", comparison: ", seltype, " ")) +
      xlab('Labels') + 
      ylab('Subgroup') +
      theme_bw() + 
      theme(plot.title = element_text(vjust=0, size=8), 
            legend.box.margin=margin(0, 0, 0, -15),
            panel.grid.major.x = element_blank(),
            panel.grid.minor.x = element_blank(),
            panel.grid.major.y = element_blank(),
            panel.grid.minor.y = element_blank(),
            panel.border = element_blank(),
            axis.line = element_line(colour = "black"),
            axis.text.x = element_text(size=4),
            axis.title = element_text(size=8),
            legend.title = element_text(size = 6), 
            legend.text  = element_text(size = 4))
    plot
    name <- paste0(selmethod, seltype, ".pdf", "")
    ggsave(name, width = 10, height = 6, units = "cm")
  }
}

# make a plot of the records in a subgroup
file <- "none.txt"
results <- readLines(file, n=(q+1))[2:(q+1)]
q <- 50
qi <- 1
result <- results[q-qi]
desc <- sub("\\ with.*", "", result)
lits <- strsplit(desc,split=" and ")
lits[[1]][1] <- sub(".*: ", "", lits[[1]][1])
subset <- mutdata
for(li in 1:length(lits[[1]])){
  #print(li)
  lit <- lits[[1]][li]
  name <- sub("=.*", "", lit)
  name <- substr(name,1,nchar(name)-2)
  value <- as.numeric(sub(".*= ", "", lit))
  sign <- tail(strsplit(sub("=.*", "", lit), split="")[[1]],n=1)
  #print(name)
  #print(value)
  #print(sign)
  if(sign == "<"){
    subset <- subset[subset[,name] <= value,]
  } else if(sign == ">"){
    subset <- subset[subset[,name] >= value,]
  }
}
# random sample of size 20
set.seed(25112021)
selection <- sample.int(n=nrow(subset), size=10, replace=FALSE)
plot <- subset[selection,votinglist] %>%
  rownames_to_column() %>%
  gather("lambda", "rank", -rowname) %>%
  mutate(rank = as.numeric(rank)) %>%
  ggplot(aes(y=rowname, x=lambda, fill=rank)) + 
  geom_tile(color="white") + 
  scale_fill_steps(low = "white", high = "#b2df8a", limits = c(0,40)) + 
  #scale_fill_gradient2(low = "white", high = "purple", 
  #                     aesthetics = "fill",
  #                     #breaks = c(0,5,10,15,20,25),
  #                     limits = c(0,40)) + 
  #metR::scale_fill_discretised() + 
  scale_x_discrete(labels = as.character(c(1:37))) + 
  scale_y_discrete(expand = c(0, 0)) +
  guides(fill = guide_colorbar(barwidth = 0.4, barheight=3)) +
  #guides(fill = guide_legend(title = "Rank")) + 
  xlab('Labels') + 
  ylab('Record') +
  theme_bw() + 
  theme(plot.title = element_text(vjust=0, size=8), 
        legend.box.margin=margin(0, 0, 0, -10),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        panel.grid.major.y = element_blank(),
        panel.grid.minor.y = element_blank(),
        panel.border = element_blank(),
        axis.line = element_line(colour = "black"),
        axis.text.x = element_text(size=4),
        axis.text.y = element_text(size=4),
        axis.title = element_text(size=6),
        legend.title = element_text(size = 4), 
        legend.text  = element_text(size = 4))
plot
name <- paste0("subgroup1noneaverage.pdf", "")
ggsave(name, width = 10, height = 3, units = "cm")
