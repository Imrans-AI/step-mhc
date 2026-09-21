RESULTS <- "results/immrep/"           # SET FIRST - HANDOFF_9 section 8
suppressPackageStartupMessages({library(ggplot2);library(dplyr);library(pROC)})

d <- read.csv(paste0(RESULTS,"immrep_scored.csv"))
keep <- d %>% group_by(peptide) %>%
        summarise(pos=sum(binder), n=n(), .groups="drop") %>%
        filter(pos>=10, pos<n) %>% arrange(desc(pos))
cat("epitopes retained (>=10 positives):", nrow(keep), "\n")

rows <- do.call(rbind, lapply(keep$peptide, function(p){
  s <- d[d$peptide==p,]
  r <- pROC::roc(s$binder, s$score, quiet=TRUE, direction="<")
  data.frame(peptide=p, fpr=1-r$specificities, tpr=r$sensitivities,
             auc=as.numeric(r$auc), pos=sum(s$binder))
}))
ov <- pROC::roc(d$binder, d$score, quiet=TRUE, direction="<")
rows <- rbind(rows, data.frame(peptide="overall", fpr=1-ov$specificities,
                               tpr=ov$sensitivities, auc=as.numeric(ov$auc), pos=sum(d$binder)))

lv <- rows %>% distinct(peptide,auc,pos) %>% arrange(peptide=="overall", desc(auc))
rows$peptide <- factor(rows$peptide, lv$peptide)
labs <- setNames(sprintf("%s (%.3f)", lv$peptide, lv$auc), lv$peptide)
pal  <- setNames(c(scales::hue_pal()(nrow(lv)-1), "grey20"), lv$peptide)

p <- ggplot(rows, aes(fpr, tpr, colour=peptide,
                      linewidth=peptide=="overall", alpha=peptide=="overall")) +
  geom_abline(slope=1, intercept=0, linetype="dashed", colour="grey65", linewidth=.3) +
  geom_line() +
  scale_linewidth_manual(values=c(`FALSE`=.45,`TRUE`=.9), guide="none") +
  scale_alpha_manual(values=c(`FALSE`=.85,`TRUE`=1), guide="none") +
  scale_colour_manual(values=pal, labels=labs, name=NULL) +
  coord_equal(xlim=c(0,1), ylim=c(0,1)) +
  labs(x="False positive rate", y="True positive rate") +
  theme_bw(base_size=8) +
  theme(panel.grid.minor=element_blank(),
        legend.position="inside", legend.position.inside=c(.72,.28),
        legend.key.size=unit(8,"pt"), legend.text=element_text(size=6),
        legend.background=element_rect(fill=alpha("white",.85), colour=NA))

ggsave("figures/Fig_immrep_roc.pdf", p, width=90, height=90, units="mm", device=cairo_pdf)
cat("wrote figures/Fig_immrep_roc.pdf\n")
print(lv, row.names=FALSE)
