from bs4 import BeautifulSoup
from clean_utils import *
import emoji
import re
"""
cp from https://github.com/wufanyou/KDD-Cup-2022-Amazon/blob/b1090653d5d725191d1779e2fe7f096ddf286133/utils/clean/clean.py
"""


def to_symbol(data):

    if type(data) == list:
        return [to_symbol(d) for d in data]
    else:
        text = data
        text = re.sub("&#34;", '"', text)
        text = re.sub("&#39;", "'", text)
        return text


def common_us_word(data):
    if type(data) == list:
        return [common_us_word(d) for d in data]
    else:
        text = data
        text = re.sub("''", '"', text)
        text = re.sub("a/c", "ac", text)
        text = re.sub("0z", "oz", text)
        text = re.sub("”|“", '"', text)
        text = re.sub("‘|′", "'", text)
        exps = re.findall("[0-9] {0,1}'", text)

        for exp in exps:
            text = text.replace(exp, exp[0] + "feet")
        exps = re.findall('[0-9] {0,1}"', text)

        for exp in exps:
            text = text.replace(exp, exp.replace('"', "inch"))

        text = re.sub("men'{0,1} {0,1}s|mens' s", "men", text)

        return text

# TODO check spell for some words


def query_clean_v1(data):

    if type(data) == list:
        return [query_clean_v1(d) for d in data]

    elif type(data) == str:
        text = data
        product_ids = re.findall("b0[0-9a-z]{8}", text)
        if product_ids:
            for i, exp in enumerate(product_ids):
                text = text.replace(exp, f"placehold{chr(97+i)}")

        exps = re.findall("[a-zA-Z]'s|s'", text)
        for exp in exps:
            text = text.replace(exp, exp[0])

        text = re.sub(
            "\(|\)|\*|---|\+|'|,|\[|\]| -|- |\. |/ |:", " ", text)  # ignore
        text = text.strip()

        exps = re.findall("[a-zA-Z]\.", text)
        for exp in exps:
            text = text.replace(exp, exp[0])

        # ! -> l for words
        exps = re.findall("![a-zA-Z]{2}", text)
        for exp in exps:
            text = text.replace(exp, exp.replace("!", "l"))

        # a/b -> a b
        exps = re.findall("[a-zA-Z]/[a-zA-Z]", text)
        for exp in exps:
            text = text.replace(exp, exp.replace("/", " "))

        # remove "
        text = re.sub('"', " ", text)

        # remove "
        text = re.sub("'", " ", text)

        # # + [sep] + [num] -> # + [num]
        exps = re.findall("# {1}[0-9]", text)
        for exp in exps:
            text = text.replace(exp, exp.replace(" ", ""))

        # remove # without
        exps = re.findall("#[a-zA-Z]", text)
        for exp in exps:
            text = text.replace(exp, exp.replace("#", ""))

        if product_ids:
            for i, exp in enumerate(product_ids):
                text = text.replace(f"placehold{chr(97+i)}", exp)

        text = text.strip()

        return text


def get_emoji_regexp():
    # Sort emoji by length to make sure multi-character emojis are
    # matched first
    emojis = sorted(emoji.EMOJI_DATA, key=len, reverse=True)
    pattern = u'(' + u'|'.join(re.escape(u) for u in emojis) + u')'
    return re.compile(pattern)


def remove_emoji(data):
    if type(data) == list:
        return [remove_emoji(d) for d in data]
    elif type(data) == str:
        return get_emoji_regexp().sub("", data)
    else:
        raise


class BaseClean:
    clean_fns = [
        "to_lower",
        "to_symbol",
        "remove_emoji",
        "clean_contractions",
        "common_us_word",
        "query_clean_v1",
        "remove_control_char",
        "remove_duplicate",
        "remove_ending_underscore",
        "remove_starting_underscore",
        "clean_multiple_form",
        "leet_clean",
    ]

    def __init__(self, clean_fns=None):
        if clean_fns:
            self.clean_fns = clean_fns

    def __call__(self, input_texts):

        if type(input_texts) == list:
            for fn in self.clean_fns:
                fn = eval(fn)
                input_texts = fn(input_texts)

        elif type(input_texts) == str:
            input_texts = [input_texts]
            input_texts = self(input_texts)
            input_texts = input_texts[0]

        return input_texts


class BertClean(BaseClean):
    clean_fns = [
        "to_symbol",
        "remove_emoji",
        "clean_contractions",
        "common_us_word",
        "query_clean_v1",
        "remove_control_char",
        "remove_duplicate",
        "remove_ending_underscore",
        "remove_starting_underscore",
        "clean_multiple_form",
        "leet_clean",
        "strip_accents",
    ]


# Function to remove tags
def remove_tags(html):

    # parse html content
    soup = BeautifulSoup(html, "html.parser")

    for data in soup(['style', 'script']):
        # Remove tags
        data.decompose()

    # return data by retrieving the tag content
    return ' '.join(soup.stripped_strings)


bertclean = BertClean()


def clean(text):
    text = remove_tags(text)
    text = bertclean(text).strip()
    return text


# item_list = [{"product_id": "B07M7V7TBW",
#               "product_title": "50 Business Envelopes, Standard Flap (Multi Color Pack, 9.5\" x 4.125\")",
#               "product_description": "<p><b>We have what it takes to satisfy your image-conscious stationery and mailing needs</b>that best fit with these envelopes because of its simple yet classic design. It features a bright cream colored, smooth and level printing surface for superior ink holdout and fast drying with a great result of vibrant full-colored prints in front and back of the envelopes, printed on a high-tech digital press. This surely stands out all the plain ones and gets the attention all you need to your recipients</p><p>These cream-colored envelopes<b>bring value and professionalism to your personal and business communications</b>with its clean and polished presentation which letting you affix printed or handwritten labels to ensure accurate addressing of your correspondence. It’s windowless and non-see through tint features that added security and protection to keep the privacy and confidentiality to your mailing content.</p><p><b>Easy sealing with its gummed flap</b>that dries non-sticky and can be reactivated after remoistening, unlike other industrial adhesive seals that are incredibly sticky and has a habit of sticking when and where they are not wanted; causing damage to the envelope itself. Less waste generated compared to a PnS envelope where you have to remove the backing.</p><p><b>Start mailing your invoices and statements with confidence and unique style</b>with these #10 envelopes that are engineered to deliver a secure and hassle-free mailing experience. It comes with a standard measurement of 4.125 x 9.5 inches that fits the smaller envelopes inside such as #9 or #6 ¾ size and holds folded letters. They are made of a high-end grade paperwhite stock of 70lb premium uncoated text in a non-glossy finish.<b>It’s not your ordinary envelopes</b>that you thought, its non-flimsy and premium quality thicker paper!</p>",
#               "product_bullet_point": "PREMIUM QUALITY - this commercial envelope has a measurement of 4 1/8 inches x 9.5 inches and constructed of high-quality text stock that features clean and professional presentation; its quality is perfect for home, personal and office use\nELEGANT and SMOOTH FINISH – it has a classic color that made with beauty and elegance that stands out from all of the other colored envelopes and uniquely matches personal or company branding; a perfect pair for stationery paper for printable and handwritten notes\nCOLORS - Pink - Green - Gray - Blue - Cream\nWINDOWLESS and TINTED - with its great opacity paper finish and windowless design, this surely guarantees your mailing contents are secured and protected; it ensures the privacy and confidentiality of your business or personal mails are in good hands\nEASY-SEAL CLOSURE - sealing comes easy-peasy with its moisture-activated gummed seal that ensures a secure closure; its strong adhesives are designed for a reliable closure to keep safe your documents in transit without compromising the professional look",
#               "product_brand": "ALLIN",
#               "product_color": "Multi Color Pack",
#               "product_cate": ["Office Products", "Office & School Supplies", "Envelopes, Mailers & Shipping Supplies", "Envelopes", "Business Envelopes"]},
#              {"product_id": "B07NZ4T2SL", "product_title": "ValBox 200 Count #8 Double Window Envelopes 3 5/8\" x 8 11/16\" Flip and Seal Double Window Security Check Envelopes- Security Tint Pattern Designed for Home Office Secure Mailing", "product_description": "\"About ValBox. </br> </br> Engaged in envelopes, paper boxes and other home and office paper products supplies, Valbox is a direct-sale store of a paper products manufacture. We advocate the love and natural concept, insisting on using recycled material in the process of production. Hope Valbox can provide you, our valued customers, the high quality products and intimate service. </br> </br> ValBox 3 5/8 x 8 11/16\"\" Quick-Seal Closure Double Window Envelopes 200 Count</br> </br> Premium quality. Protect to fit QuickBooks and computer checks perfectly!</br> </br> Window line up perfectly.</br> </br> Bottom window location of 3/4\"\" from the bottom, and a window height of 7/8\"\" prevents the check amounts from showing in the bottom window like other check envelopes.</br> </br> Bottom window location of 7/8\"\" from the left side, and a window width of 3.5\"\" makes your recipients address will show without the need to reformat quickbooks for tri-fold, voucher, and 3 checks per page check paper. Just print and slide the check in.</br> </br> Protect your check security.</br> </br> 24 LB White wove paper, security tint design and stronger glue. Your checks will be secure with an easy to use and being a professional look from the box all the way to your destination.\"",
#                  "product_bullet_point": "Security Check Envelopes. Fits QuickBooks Checks and other computer printed checks perfectly. Fits Peachtree checks as well as Quicken and QuickBooks. The checks are easy to slip in.\nDouble Windows Design. The windows line up perfectly with return and sender address, without showing the memo or check amount line like other envelopes. No show of memo, Payee line or amount in windows. NOTICE: THESE #8 DOUBLE WINDOW ENVELOPES ARE NOT COMPATIBLE WITH QuickBooks INVOICES.\nPremium Quality. Our Valbox double window envelopes feature with durable 24lbs white wove paper, peel-off strip, stronger glue and security tint design. Protect your check privacy.\nGreat Self Seal Envelopes. The self-seal feature makes these envelopes much more convenient than regular, moisten-and-seal ones. You don't need to stand licking envelopes and using glue or the water activators, which is time consuming and messy.\nStrong Adhesive: With the strong adhesive flap, ValBox #8 double window envelopes will be more evident if tampered. No need to worry about opening them easily by others. If there's anything make you unhappy or unsatisfied about the products, please do contact us without any hesitation.", "product_brand": "ValBox", "product_color": "200 Count", "product_cate": ["Office Products", "Office & School Supplies", "Envelopes, Mailers & Shipping Supplies", "Envelopes", "Business Envelopes"]}
#              ]
# clean('aluminum foil - silver')
