#!/usr/bin/env Rscript
# Run the notebook's CUSTOM Farrington implementation on Python-pipeline TEST sims.
# The custom function defs (algo.farrington + helpers) are inlined below so
# this script has NO external file dependencies.
# Returns BOTH $alarm (with limit54 filter) and $alarmall (without filter).
#
# Saves per signal:
#   farrington_custom_alarms_signal_{S}.csv    (with limit54)
#   farrington_custom_alarmsall_signal_{S}.csv (without limit54)

suppressPackageStartupMessages({
  if (!require(surveillance, quietly = TRUE)) install.packages("surveillance", repos="https://cran.rstudio.com/")
  library(surveillance)
  if (!require(jsonlite, quietly = TRUE)) install.packages("jsonlite", repos="https://cran.rstudio.com/")
  library(jsonlite)
})

# ============= INLINED CUSTOM FARRINGTON FUNCTIONS =============
algo.farrington=function (disProgObj, control = list(range = NULL, b = 3, w = 3,
                                                     reweight = TRUE, verbose = FALSE, alpha = 0.01, trend = TRUE,
                                                     limit54 = c(5, 4), powertrans = "2/3", fitFun = "algo.farrington.fitGLM"))
{
  observed <- disProgObj$observed

  freq <- disProgObj$freq
  #epochStr <- switch(as.character(freq), `12` = "1 month",
  #    `52` = "1 week", `365` = "1 day")
  if (is.null(control$range)) {
    control$range <- (freq * control$b - control$w):length(observed)
  }
  if (is.null(control$b)) {
    control$b = 5
  }
  if (is.null(control$w)) {
    control$w = 3
  }
  if (is.null(control$reweight)) {
    control$reweight = TRUE
  }
  if (is.null(control$verbose)) {
    control$verbose = FALSE
  }
  if (is.null(control$alpha)) {
    control$alpha = 0.01
  }
  if (is.null(control$trend)) {
    control$trend = TRUE
  }
  if (is.null(control$plot)) {
    control$plot = FALSE
  }
  if (is.null(control$limit54)) {
    control$limit54 = c(5, 4)
  }
  if (is.null(control$powertrans)) {
    control$powertrans = "2/3"
  }
  if (is.null(control$fitFun)) {
    control$fitFun = "algo.farrington.fitGLM"
  }
  #else {
  #    control$fitFun <- match.arg(control$fitFun, c("algo.farrington.fitGLM"))
  #}
  #if (is.null(disProgObj[["epochAsDate", exact = TRUE]])) {
  #    epochAsDate <- FALSE
  #}
  #else {
  #    epochAsDate <- disProgObj[["epochAsDate", exact = TRUE]]
  #}
  if (!((control$limit54[1] >= 0) & (control$limit54[2] > 0))) {
    stop("The limit54 arguments are out of bounds: cases >= 0 and period > 0.")
  }
  # alarmall gives the results for all data sets whereas alarm identifies the
  # ones where there are less than 5 reports in the last 4 weeks
  alarmall <- matrix(data = 0, nrow = length(control$range), ncol = 1)
  alarm<- matrix(data = 0, nrow = length(control$range), ncol = 1)
  muhat <- matrix(data = NA_real_, nrow = length(control$range), ncol = 1)  # per-day Farrington baseline mu-hat
  upperbound <- matrix(data = 0, nrow = length(control$range), ncol = 1)
  trend <- matrix(data = 0, nrow = length(control$range), ncol = 1)
  pd <- matrix(data = 0, nrow = length(control$range), ncol = 2)

  n <- control$b * (2 * control$w + 1)

  # identify the leading zeroes in the data and cut them off

  #observed1=observed[observed!=NA]

  for (k in control$range) {
    #observed<-rollapply(data=observed[1:k], width=7, FUN=sum, ascending = F)
    #observed<-rollapply(data=a, width=7, FUN=sum, ascending = F)
    observed=observed[1:k]
    ob=rep(0,round(length(observed)/7))
    for(w in 0:(round(length(observed)/7)-1)){
      if((k-6*w-6-w)<=0){break}
      ob[w+1]=sum(observed[(k-6*w-w):(k-6*w-6-w)])
    }
    observed=ob[length(ob):1]
    kk=length(observed)
    for(z in 1:length(observed)){
      if(observed[z]==0){observed[z]=NA}
      else{break}
    }
    if (control$verbose) {
      cat("k=", k, "\n")
    }
    # If the remaining data is less than one year's worth,
    # do not fit a model and indicate that by reporting the value 1e+300
    if((kk-z)<(2*round(freq)+control$w+1)){
      upperbound[k - min(control$range) + 1] = 1e+300
      alarmall[k - min(control$range) + 1] = 1e+300
      alarm[k - min(control$range) + 1] = 1e+300
      trend[k - min(control$range) + 1] = 1e+300
    }
    else{
      #if (!epochAsDate) {
      # Assign level factors to the data (I have done that manually given
      # that the data can be missing...)
      seas1 <- NULL
      seas2 <- NULL
      for (i in control$b:0) {
        if(i==3){seas1=append(seas1, seq(kk - round(freq * i) -
                                           control$w, kk - round(freq * i) +control$w+1, by = 1))}
        else{seas1=append(seas1, seq(kk - round(freq * i) -
                                       control$w, kk - round(freq * i) +control$w, by = 1))}
        seas2=append(seas2, seq(kk - round(freq * i) -
                                  control$w-5, kk - round(freq * i) -control$w-1, by = 1))
      }
      #}
      seas3=seas2-5
      seas4=seas3-5
      seas5=seas4-5
      seas6=seas5-5
      seas7=seas6-5
      seas8=seas7-5
      seas9=seas8-5
      seas10=seas9-5
      seasgroup=rep(0,(length(observed)+control$w))
      seasgroup[seas1]=1
      seasgroup[seas2]=2
      seasgroup[seas3]=3
      seasgroup[seas4]=4
      seasgroup[seas5]=5
      seasgroup[seas6]=6
      seasgroup[seas7]=7
      seasgroup[seas8]=8
      seasgroup[seas9]=9
      seasgroup[seas10]=10
      if((kk-z)<(freq*control$b+control$w+1)){
        seasgroup=seasgroup[z:(kk-27)]
        seasgroup=as.factor(seasgroup)
        wtime= z:(kk-27)
      }
      else{
        seasgroup=seasgroup[(kk-control$b*freq-control$w-1):(kk-27)]
        seasgroup=as.factor(seasgroup)
        wtime= (kk-control$b*freq-control$w-1):(kk-27)
      }
      response <- observed[wtime]
      if (control$verbose) {
        print(response)
      }

      v=length(response[response>0])
      toosmall=(v<2) # indicator for when (stricktly) less than 2 weeks have non-zero counts

      # If stricktly less than 2 weeks have non-zero counts,
      # do not fit a model and indicate that by reporting the value 1e+400

      if(toosmall){
        upperbound[k - min(control$range) + 1] = 1e+400
        alarmall[k - min(control$range) + 1] = 1e+400
        alarm[k - min(control$range) + 1] = 1e+400
        trend[k - min(control$range) + 1] = 1e+400
      }
      else{
        oneyear=FALSE
        p=wtime[response>0]
        oneyear=((p[length(p)]-p[1])<52)
        # if all non-zero weeks are within one year, do not fit a trend.
        # This is the only case where we do not fit a trend.
        if(oneyear){
          model <- do.call(control$fitFun, args = list(response = response,
                                                       wtime = wtime, seasgroup=seasgroup,timeTrend = FALSE, reweight = control$reweight))
        }
        else{
          model <- do.call(control$fitFun, args = list(response = response,
                                                       wtime = wtime, seasgroup=seasgroup,timeTrend = control$trend, reweight = control$reweight))
        }
        if(is.null(model)){return(model)}

        doTrend <- control$trend
        if (model$T==1) {
          wp <- summary.glm(model)$coefficients["wtime", 4] # p-value after reweighting
          # In our new approach, we fit a trend always (unless all non-zero weeks are within one year).
          # For this reason, we consider the trend to be always significant (wp<=1).
          # If one wants to fit a trend only if significant, change this to, say, wp<=0.05.
          significant <- (wp <=1)
          mu0Hat <- predict.glm(model, data.frame(wtime = c(kk),seasgroup=factor(1)),type = "response")
          #atLeastThreeYears <- (control$b >= 3)
          # We remove the noExtrapolation condition
          #noExtrapolation <- mu0Hat <= max(response)
          if (!(significant)) {
            doTrend <- FALSE
            model <- do.call(control$fitFun, args = list(response = response,
                                                         wtime = wtime,seasgroup=seasgroup, timeTrend = FALSE, reweight = control$reweight))
          }
        }
        else {
          doTrend <- FALSE
        }
        if(model$phi<1){model$phi=1}
        pred <- predict.glm(model, data.frame(wtime = c(kk),seasgroup=factor(1)),
                            dispersion = model$phi, type = "response", se.fit = TRUE)
        # We use negative binomial quantiles to define the error structure rather
        # than the ones based on the transformed Poisson and the Anscombe residuals.
        if(model$phi==1){
          lu<-c(qpois(control$alpha,pred$fit),qpois(1-control$alpha,pred$fit))
          lu[2]=max(1,lu[2])
        }
        else{
          lu<-c(qnbinom(control$alpha,pred$fit/(model$phi-1),1/model$phi),qnbinom(1-control$alpha,pred$fit/(model$phi-1),1/model$phi))
          lu[2]=max(1,lu[2])
        }
        #if (control$plot) {
        #        data <- data.frame(wtime = seq(min(wtime), k, length = 1000))
        #        preds <- predict(model, data, type = "response",
        #            dispersion = model$phi)
        #        plot(c(wtime, k), c(response, observed[k]), ylim = range(c(observed[data$wtime],
        #            lu)), , xlab = "time", ylab = "No. infected",
        #            main = paste("Prediction at time t=", k, " with b=",
        #              control$b, ",w=", control$w, sep = ""), pch = c(rep(1,
        #              length(wtime)), 16))
        #        lines(data$wtime, preds, col = 1, pch = 2)
        #        lines(rep(k, 2), lu[1:2], col = 3, lty = 2)
        #    }

        enoughCases <- (sum(observed[(kk - control$limit54[2] + 1):kk]) >= control$limit54[1])

        X <- (observed[kk] - pred$fit)/(lu[2] - pred$fit)
        upperbound[k - min(control$range) + 1] <- lu[2]
        muhat[k - min(control$range) + 1] <- pred$fit
        alarm[k - min(control$range) + 1] <- (X > 1)
        # if the last five weeks have less than 4 reports,
        # indicate that by returning the value 1e-300
        if(!enoughCases){alarm[k - min(control$range) + 1] = 1e-300}
        alarmall[k - min(control$range) + 1] <- (X > 1)
        trend[k - min(control$range) + 1] <- doTrend
      }
    }
    observed <- disProgObj$observed
  }
  control$name <- paste("farrington(", control$w, ",", 0, ",", control$b, ")", sep = "")
  control$data <- paste(deparse(substitute(disProgObj)))
  result <- list(alarm = alarm, alarmall = alarmall, trend = trend,
                 disProgObj = disProgObj, control = control,upperbound=upperbound, muhat=muhat)
  class(result) <- "survRes"
  return(result)
}

#=========================================

algo.farrington.fitGLM=function (response, wtime,seasgroup, timeTrend = TRUE, reweight = TRUE)
{
  theModel <- as.formula(ifelse(timeTrend, "response~seasgroup+wtime",
                                "response~seasgroup"))
  model <- glm(theModel, family = quasipoisson(link = "log"))
  # In our new approach, we fit a trend always (unless all non-zero weeks are within one year).
  # For this reason, we consider the trend always significant (p<=1).
  # If one wants to fit a trend only if significant, change this to, say, p<=0.05.
  if (model$converged){
    if (timeTrend) {
      p<- summary.glm(model)$coefficients["wtime", 4] # p-value before reweighting
      if(p<=1){T=1}
      else{
        T=0
        model <- glm(response ~ seasgroup, family = quasipoisson(link = "log"))
        if(!model$converged) {
          cat("Warning: No convergence without insignificant timeTrend.\n")
          print(cbind(response, wtime))
          return(NULL)
        }
      }
    }
    if(!timeTrend){T=0}
  }

  else {
    T=0
    if (timeTrend) {
      model <- glm(response ~ seasgroup, family = quasipoisson(link = "log"))
      cat("Warning: No convergence with timeTrend -- trying without.\n")
    }
    if (!model$converged) {
      cat("Warning: No convergence without timeTrend.\n")
      print(cbind(response, wtime))
      return(NULL)
    }
  }

  phi <- max(summary(model)$dispersion, 1)
  if (reweight) {
    s <- anscombe.residuals(model, phi)
    omega <- algo.farrington.assign.weights(s)
    theModel <- as.formula(ifelse(T==1, "response~seasgroup+wtime", "response~seasgroup"))
    model <- glm(theModel, family = quasipoisson(link = "log"),weights = omega)
    if (!model$converged) {

      if (T==1) {
        model <- glm(response ~ seasgroup, family = quasipoisson(link = "log"),weights = omega)
        cat("Warning: No convergence with weights and timeTrend -- trying without.\n")
      }
      if (!model$converged) {
        cat("Warning: No convergence with weights but without timeTrend.\n")
        print(cbind(response, wtime))
        return(NULL)
      }
      T=0
    }

    phi <- max(summary(model)$dispersion, 1)
  }
  model$phi <- phi
  model$T=T
  return(model)
}

#=========================================


algo.farrington.assign.weights=function (s)
{   # we use the cut off point s>2.58 as a compromise between s>2 and s>3
  gamma <- length(s)/(sum((s^(-2))^(s > 2.58)))
  omega <- numeric(length(s))
  omega[s > 2.58] <- gamma * (s[s > 2.58]^(-2))
  omega[s <= 2.58] <- gamma
  return(omega)
}


#=========================================


anscombe.residuals=function (m, phi)
{
  y <- m$y
  mu <- fitted.values(m)
  a <- 3/2 * (y^(2/3) * mu^(-1/6) - mu^(1/2))
  a <- a/sqrt(phi * (1 - hatvalues(m)))
  return(a)
}
# ============= END INLINED FUNCTIONS =============

DATA_DIR <- Sys.getenv("SYND_DATA_DIR", "big_signal_datasets_small")
DAYS <- 7
VALID_DAYS <- 49 * 7   # 343

splits <- fromJSON("splits_for_r.json")
signals <- if (nzchar(Sys.getenv("SYND_SIGNALS"))) as.integer(strsplit(Sys.getenv("SYND_SIGNALS"), ",")[[1]]) else 1:16
args <- commandArgs(trailingOnly = TRUE)
alpha_val <- 0.01
if (length(args) >= 1) alpha_val <- as.numeric(args[1])
# Second arg selects the split to evaluate: "test" (default) or "val".
# The val split lets us re-select ensemble configurations on validation data
# (avoiding selection-on-test bias) before reporting on the test split.
split_name <- "test"
if (length(args) >= 2) split_name <- args[2]
stopifnot(split_name %in% c("test", "val"))
base_suffix <- if (alpha_val == 0.01) "custom" else sprintf("custom_a%03d", round(alpha_val * 1000))
out_suffix <- if (split_name == "val") paste0(base_suffix, "_val") else base_suffix
cat(sprintf("Using alpha=%.3f; split='%s'; output suffix='%s'; DATA_DIR=%s\n",
            alpha_val, split_name, out_suffix, DATA_DIR))

for (S in signals) {
  cat(sprintf("\n=== Signal %d (Farrington-CUSTOM, alpha=%.3f) ===\n", S, alpha_val))
  totals_fp    <- file.path(DATA_DIR, sprintf("simulated_totals_sig%d.csv", S))
  outbreaks_fp <- file.path(DATA_DIR, sprintf("simulated_outbreaks_sig%d.csv", S))
  if (!file.exists(totals_fp)) { cat("  missing totals; skip\n"); next }
  totals <- read.csv(totals_fp, check.names = FALSE)
  outbreaks_raw <- read.csv(outbreaks_fp, check.names = FALSE)
  outbreaks <- as.matrix((outbreaks_raw > 0) * 1)
  N <- nrow(totals)
  weeks <- N / DAYS

  test_idx0 <- splits[[paste0("signal_", S)]][[split_name]]
  test_idx1 <- test_idx0 + 1L
  n_test <- length(test_idx1)
  cat(sprintf("  %d %s sims; N=%d days; weeks=%.2f\n", n_test, split_name, N, weeks))

  alarm_mat    <- matrix(0L, nrow = VALID_DAYS, ncol = n_test)
  alarmall_mat <- matrix(0L, nrow = VALID_DAYS, ncol = n_test)
  out_mat      <- matrix(0L, nrow = VALID_DAYS, ncol = n_test)
  muhat_mat    <- matrix(NA_real_, nrow = VALID_DAYS, ncol = n_test)  # Farrington baseline mu-hat per eval day

  cntrl <- list(range = (N - VALID_DAYS + 1):N,
                w = 3, b = 5, alpha = alpha_val, trend = TRUE,
                fitFun = "algo.farrington.fitGLM")

  t0 <- Sys.time()
  for (j in seq_len(n_test)) {
    i <- test_idx1[j]
    a.disProg <- create.disProg(week = weeks,
                                observed = totals[, i],
                                freq = 52.18)
    a.f <- algo.farrington(a.disProg, control = cntrl)
    av  <- as.integer(as.vector(a.f$alarm))
    aav <- as.integer(as.vector(a.f$alarmall))
    if (length(av) != VALID_DAYS || length(aav) != VALID_DAYS) {
      stop(sprintf("Signal %d sim %d: length mismatch (alarm=%d alarmall=%d)",
                   S, i, length(av), length(aav)))
    }
    alarm_mat[, j]    <- av
    alarmall_mat[, j] <- aav
    out_mat[, j]      <- as.integer(outbreaks[(N - VALID_DAYS + 1):N, i])
    muhat_mat[, j]    <- as.numeric(as.vector(a.f$muhat))
    if (j %% 25 == 0) {
      cat(sprintf("  sim %d/%d done (%.1fs elapsed)\n", j, n_test,
                  as.numeric(difftime(Sys.time(), t0, units = "secs"))))
    }
  }
  dt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  cat(sprintf("  done in %.1fs\n", dt))

  colnames(alarm_mat)    <- paste0("sim_", seq_len(n_test) - 1)
  colnames(alarmall_mat) <- paste0("sim_", seq_len(n_test) - 1)
  colnames(out_mat)      <- paste0("sim_", seq_len(n_test) - 1)
  write.csv(alarm_mat,    file = sprintf("farrington_%s_alarms_signal_%d.csv",    out_suffix, S), row.names = FALSE)
  write.csv(alarmall_mat, file = sprintf("farrington_%s_alarmsall_signal_%d.csv", out_suffix, S), row.names = FALSE)
  write.csv(out_mat,      file = sprintf("farrington_%s_outbreaks_signal_%d.csv", out_suffix, S), row.names = FALSE)
  colnames(muhat_mat) <- paste0("sim_", test_idx0)   # keyed by ORIGINAL sim index for Python alignment
  write.csv(muhat_mat,    file = sprintf("farr_baseline_%s_signal_%d.csv", out_suffix, S), row.names = FALSE)
  cat(sprintf("  saved (alarm/alarmsall/outbreaks/baseline)\n"))
}

cat("\nALL DONE.\n")
