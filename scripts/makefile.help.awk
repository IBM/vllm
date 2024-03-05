# to start, initialize some variables
BEGIN {
    # set the field separator to split a target from its help string
    FS = ":.*##";
    # const variables
    common_target_suffix = ".common_scripts"
    null_category = ""
    target_help_key = "help_string"
    target_category_key = "category"
    # track the current category
    current_category = null_category;
    # the set of categories
    meaningless_value = 0
    categories_set[null_category] = meaningless_value
    # and the order they are defined
    categories_index = 1;
    categories_order[categories_index] = current_category;
}

# Matches a target with a help string
/^[-a-zA-Z_0-9\\.]+:.*?##/ {
    target = $1
    # remove target suffix if it exists
    regex = sprintf("%s$", common_target_suffix)
    sub(regex, "", target)
    if (!(target in targets)) {
        targets[target][target_category_key] = current_category;
        targets[target][target_help_key] = sprintf("  \033[36m%-15s\033[0m %s", target, $2)
    }
}

# Updates the category for subsequent targets
/^##@/ {
    current_category = sprintf("\n\033[1m%s\033[0m", substr($0, 5));
    if (! (current_category in categories_set)) {
        categories_set[current_category] = meaningless_value
        categories_index++;
        categories_order[categories_index] = current_category;
    }
}

# Reset the category to allow parsing multiple files
ENDFILE {current_category=null_category}

END {
    # build the categories -> targets map
    for (t in targets) {
        tcat = targets[t][target_category_key]
        thelp = targets[t][target_help_key]
        categories_map[tcat][t] = thelp
    }

    # print the help output
    printf("\nUsage:\n  make \033[36m<target>\033[0m\n ")

    for (ci=1; ci<=length(categories_order); ci++) {
        c = categories_order[ci]
        if (c in categories_map) {
            print c;
            tn = asort(categories_map[c], ts);
            for (ti=1; ti<=tn; ti++)
                print ts[ti];
        }
    }
}
