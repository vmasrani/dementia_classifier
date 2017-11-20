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
    general_plots.vanilla_feature_set_plot()
    general_plots.plot_feature_selection_curve()
    general_plots.plot_feature_rank('none')
    general_plots.plot_feature_rank('halves')

    feature_set.ablation_plot(metric='fms')
    feature_set.new_feature_set_plot(metric='fms', absolute=True)
    feature_set.new_feature_set_plot(metric='fms', absolute=False)

    domain_adapt.good_classifiers_plot(metric='fms')
    domain_adapt.bad_classifiers_plot(metric='fms')
    
    blog.blog_plot()
    blog.plot_blog_feature_selection_curve()
    blog.feature_box_plot('getAoaScore')
    blog.feature_box_plot('getConcretenessScore')
    blog.feature_box_plot('getImagabilityScore')
    blog.feature_box_plot('getSUBTLWordScores')


def main():
    save_features_to_database()
    save_all_results()
    save_all_plots()
    
main()
