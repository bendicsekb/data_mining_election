library(ggplot2)
library(mice)
library(tidyverse)

# Working directory
setwd("C:/Users/20200059/Documents/Github/data_mining_election/Synthetic/")

data <- read.csv("reversed.csv")
names(data)

pd <- position_dodge(.65)
plot <- data %>%
  rename("rows" = names(data)[1]) %>%
  gather("measure", "time", -c(rows, descs, targets)) %>%
  mutate(rows = as.factor(rows)) %>%
  mutate(targets = as.factor(targets)) %>%
  mutate(descs = as.factor(descs)) %>%
  mutate(measure = ordered(measure,
                           levels = c("entropy", "sqrtnN", "norm", "labelwise", "pairwise"))) %>%
  #filter(targets == 32) %>%
  filter(descs == 32) %>%
  filter(measure %in% c('entropy','norm')) %>%
  ggplot(aes(x=measure, y=time, color=rows, shape=targets)) +
  geom_point(position = pd) + 
  ylab('Runtime in seconds') + 
  xlab('Quality measure') +
  #labs(title = '') + 
  theme_bw() + 
  theme(plot.title = element_text(vjust=-4), 
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        #panel.grid.major.y = element_blank(),
        panel.grid.minor.y = element_blank(),
        panel.border = element_blank(),
        axis.line = element_line(colour = "black"),
        axis.text.x = element_text(size=4),
        axis.title = element_text(size=10),
        legend.title = element_text(size = 6), 
        legend.text  = element_text(size = 4),
        #legend.position="bottom",
        #legend.justification="right",
        #legend.margin=margin(-20, 0, 0, 0))
  )
plot
name <- paste('timereversed.pdf', sep = "", collapse = NULL)
ggsave(name, width = 12, height = 10, units = "cm")

