import logging

def generate_mutant_sequence(aa_wt, variant, offset=0):
    variants = variant.split(",")
    aa_mut = list(aa_wt)
    for v in variants:
        pos = int(v[1:-1]) + offset
        old_aa = v[0]
        new_aa = v[-1]
        if aa_mut[pos] != old_aa:
            logging.info("Error: Position %d is not %s in the wildtype sequence" % (pos, old_aa))
            return
        aa_mut[pos] = new_aa

    return "".join(aa_mut)