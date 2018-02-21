from dementia_classifier.feature_extraction import save_dbank_to_sql, save_blog_to_sql
from dementia_classifier.analysis import feature_set, domain_adapt, blog, general_plots


def save_features_to_database():
    save_blog_to_sql.save_blog_data()
    save_dbank_to_sql.save_all_to_sql()


def save_all_results():
    print "-----------------------------------------"
    print "Saving: save_new_feature_results_to_sql()"
    print "-----------------------------------------"
    feature_set.save_new_feature_results_to_sql()

    print "-----------------------------------------"
    print "Saving: save_ablation_results_to_sql()"
    print "-----------------------------------------"
    feature_set.save_ablation_results_to_sql()

    print "-----------------------------------------"
    print "Saving: save_domain_adapt_results_to_sql()"
    print "-----------------------------------------"
    domain_adapt.save_domain_adapt_results_to_sql()

    print "-----------------------------------------"
    print "Saving: save_blog_results_to_sql()"
    print "-----------------------------------------"
    blog.save_blog_results_to_sql()

    print "-----------------------------------------"
    print "Saving: save_blog_ablation_results_to_sql()"
    print "-----------------------------------------"
    blog.save_blog_ablation_results_to_sql()


def save_all_plots():
    # Baselines
    general_plots.plot_feature_selection_curve()
    general_plots.vanilla_feature_set_plot()
    feature_set.ablation_plot(metric="fms")
    general_plots.plot_feature_rank('none')

    # New features
    feature_set.new_feature_set_plot(metric='fms', absolute=True,  poly=False)
    feature_set.new_feature_set_plot(metric='fms', absolute=True,  poly=True)
    feature_set.new_feature_set_plot(metric='fms', absolute=False, poly=True)
    feature_set.new_feature_set_plot(metric='fms', absolute=False, poly=False)
    general_plots.plot_feature_rank('halves')

    # New feature boxplots
    feature_set.feature_box_plot("MeanWordLength")
    feature_set.feature_box_plot("NP_to_PRP")
    feature_set.feature_box_plot("age")
    feature_set.feature_box_plot("prcnt_rs_uttered")
    feature_set.feature_box_plot("getImagabilityScore")

    # New feature appendix
    feature_set.new_feature_set_plot(metric='fms', absolute=False, poly=False)
    feature_set.new_feature_set_plot(metric='acc', absolute=True)
    feature_set.new_feature_set_plot(metric='acc', absolute=False, poly=True)
    feature_set.new_feature_set_plot(metric='roc', absolute=True)
    feature_set.new_feature_set_plot(metric='roc', absolute=False, poly=True)
    feature_set.ablation_plot(metric="acc")
    feature_set.ablation_plot(metric="roc")

    # Domain adaptation
    domain_adapt.good_classifiers_plot(metric='fms')
    domain_adapt.bad_classifiers_plot(metric='fms')

    # Domain adaptation appendix
    domain_adapt.good_classifiers_plot(metric='acc')
    domain_adapt.bad_classifiers_plot(metric='acc')

    # Blog
    blog.plot_blog_feature_selection_curve(metric='roc')
    blog.blog_plot()
    blog.blog_ablation_plot()
    blog.plot_blog_feature_rank()

    # Blog boxplot
    blog.blog_feature_box_plot('S')
    blog.blog_feature_box_plot('NP_to_PRP')
    blog.blog_feature_box_plot('MeanWordLength')
    blog.blog_feature_box_plot('getSUBTLWordScores')


def main():
    save_features_to_database()
    save_all_results()
    save_all_plots()


if __name__ == '__main__':
    main()
