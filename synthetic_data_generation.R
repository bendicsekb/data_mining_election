# Working directory
setwd("C:/Users/etc")

# Compute the list of averages that will be used for random target generation
averages <- function(nr_columns) {
  if (nr_columns < 1) {
    stop("[!] The number of target variables is set to ", nr_columns, 
         ", but should be at least 1.")
  }
  
  avgs <- c()
  avgs[1] <- 1
  for (i in 2:nr_columns) {
    avgs[i] <- avgs[i-1] * 0.7
  }
  
  return(avgs)
}

# Compute the list of variances that will be used for random target generation
variances <- function(nr_columns) {
  vars <- 0.03 * averages(nr_columns) 
  return(vars)
}

# Generate the target variables for one row
random_target_row <- function(nr_columns) {
  avgs <- averages(nr_columns)
  vars <- variances(nr_columns)
  
  # Generate random vote distribution
  raw_data <- c()
  for(i in 1:nr_columns) {
    raw_data[i] <- rnorm(1, mean=avgs[i], sd=sqrt(vars[i]))
  }
  
  # Compute rank, given the vote distribution
  sorted_data <- sort(raw_data, decreasing = TRUE)
  rank_data <- match(raw_data, sorted_data)
  
  return(rank_data)
}

random_descriptor_row <- function(nr_columns) {
  data <- runif(nr_columns, min=0, max=1) > 0.5
  return(data)
}

synthetic_data <- function(nr_rows, nr_descriptors, nr_target_vars,
                                    subgroup_target_function) {
  nr_columns <- nr_descriptors + nr_target_vars
  df <- data.frame(matrix(NA, nrow = nr_rows, ncol = nr_columns))
  subgroup_row <- subgroup_target_function(nr_target_vars)
  
  # Data generation
  for (i in 1:nr_rows) {
    descriptors <- random_descriptor_row(nr_descriptors)
    targets <- c()
    if (descriptors[1] == 1 && descriptors[2] == 1) {     # If in real subgroup
      targets <- subgroup_row
    } else {
      targets <- random_target_row(nr_target_vars)
    }
    row <- c(descriptors, targets)
    df[i,] <- row
  }
  
  # Column names
  descriptor_names <- c()
  for (i in 1:nr_descriptors) {
    descriptor_names[i] <- paste("descriptor", i, sep="")
  }
  
  target_var_names <- c()
  for (i in 1:nr_target_vars) {
    target_var_names[i] <- paste("party", i, sep="")
  }
  
  names(df) <- c(descriptor_names, target_var_names)
  
  return(df)
}

reverse_targets <- function(nr_target_vars) {
  targets <- c()
  for (i in 1:nr_target_vars) {
    targets[i] <- nr_target_vars - i + 1
  }
  return(targets)
}

pairwise_swapped_targets <- function(nr_targets) {
  targets <- c()
  for (i in 1:nr_targets) {
    odd <- (i %% 2) == 1
    if (odd && i == nr_targets) {
      targets[i] <- i
    } else if(odd) {
      targets[i] <- i+1
    } else {
      targets[i] <- i-1
    }
  }
  return(targets)
}

ordered_targets_with_different_first <- function(nr_targets, new_first_place) {
  targets <- c()
  targets[1] <- new_first_place
  for (i in 2:nr_targets) {
    if (i <= new_first_place) {
      targets[i] <- i-1
    } else {
      targets[i] <- i
    }
  }
  return(targets)
}

targets_second_to_first <- function(nr_targets) {
  return(ordered_targets_with_different_first(nr_targets, 2))
}

targets_middle_to_first <- function(nr_targets) {
  return(ordered_targets_with_different_first(nr_targets, nr_targets%/%2))
}

targets_last_to_first <- function(nr_targets) {
  return(ordered_targets_with_different_first(nr_targets, nr_targets))
}

##### Example queries
subgroup_type_folder_name <- "type1/"
subgroup_type <- reverse_targets

subgroup_types <- c(reverse_targets, pairwise_swapped_targets, targets_last_to_first)
subgroup_type_folders <- c("reversed", "pairwise_swapped", "last_to_first")

for (type in 1:length(subgroup_types)) {
  subgroup_type <- subgroup_types[type][1]
  subgroup_type_folder_name <- subgroup_type_folders[type]
    print("")
  print("")
  print(subgroup_type_folder_name)
  for (nr_rows in c(1000, 500, 100)) {
    print(paste("nr_rows =", nr_rows))
    for (nr_descriptors in c(32, 8, 2)) {
      print(paste("    nr_descriptors =", nr_descriptors))
      for (nr_target_vars in c(32, 8, 2)) {
        folder <- paste(
          subgroup_type_folder_name, "/",
          "nrow", nr_rows, 
          "_ndescr", nr_descriptors,
          "_ntarget", nr_target_vars, sep="")
        dir.create(subgroup_type_folder_name, showWarnings = FALSE)
        dir.create(folder, showWarnings = FALSE)
        for (i in 1:50) {
          synthetic <- synthetic_data(nr_rows, nr_descriptors, nr_target_vars, subgroup_type[[1]])
          write.csv(x=synthetic, file=paste(folder, "/data", i, ".csv", sep=""))
        }
      }
    }
  }
}
