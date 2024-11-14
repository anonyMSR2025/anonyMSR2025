import bisect

def get_offset(x, sizes_cumsum, this_row):
    bin_id = bisect.bisect_right(sizes_cumsum, x) - 1
    offset = x - sizes_cumsum[bin_id]
    # deepcopys[i][bin_id][offset] = 3
    if bin_id == 0:
        return "text", -1, offset
    else:
        sorted_ids = this_row[1]
        original_bin_id = sorted_ids[bin_id - 1]
        return "code", original_bin_id, offset