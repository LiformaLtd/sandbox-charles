"""New authored prompt families; variations supplement varied ARCTIC sentences."""
FAMILIES = {
 'closure-m': [
 ('My mother made {} before the morning meeting.', ['a warm meal','some marmalade','a mug of milk','a mushroom pie','a small menu','a mint pudding','a bowl of cream','some lemon jam']),
 ('We remembered {} when the music began.', ['the missing map','a moment at home','my first summer','the morning mist','the family meal','our meeting in May','the name of the museum','a message from Mum']),
 ('The manager mentioned {} in the message.', ['a small improvement','the monthly meeting','some missing money','a memorable moment','the summer programme','my membership number','the museum opening','a problem with the microphone']),
 ('Please move {} away from the microphone.', ['my mug','the small mirror','the menu','the measuring tape','my morning meal','the metal frame','the empty hamper','the camera']),
 ],
 'closure-b': [
 ('Ben brought {} back before breakfast.', ['a blue blanket','both brown boxes','a bag of bread','the borrowed book','a bunch of berries','the broken bottle','a bright blue ball','the bedroom brush']),
 ('Barbara bought {} beside the bridge.', ['a big basket','some brown bread','a blue bucket','a beautiful bracelet','a bottle of bubbles','a bag of beans','a black belt','a bunch of bananas']),
 ('The builder balanced {} on the bench.', ['a wooden board','a brown bucket','a box of bolts','a blue bottle','the broken beam','a bag of bricks','the broad brush','a book of plans']),
 ('Both boys believed {} would be better.', ['the blue boat','a bigger bedroom','the brown bench','a bright banner','a bowl of berries','the back balcony','a broad bridge','a break before bed']),
 ],
 'closure-p': [
 ('Please put {} beside the paper plates.', ['the purple pen','a pair of slippers','the spare cup','a packet of peas','the pink pillow','the printed map','a pot of soup','the picture book']),
 ('Peter packed {} before the picnic.', ['a paper cup','some ripe peaches','a pink blanket','a packet of chips','a pair of boots','the purple backpack','a pile of plates','a spare sweater']),
 ('The painter picked up {} after the pause.', ['a pot of paint','the paper pattern','a purple pencil','a piece of paper','the spare palette','the printed poster','a plastic cup','the pink paintbrush']),
 ('We stopped to pick up {} on the path.', ['a pretty pebble','a paper parcel','the dropped pen','a pair of gloves','a purple flower','the picnic basket','a piece of rope','a small package']),
 ],
 'contrast': [
 ('Mum asked Ben to pack {} for the boat.', ['a warm blanket','some bread and butter','the paper map','a bottle of milk','a pair of slippers','the blue backpack','a small pump','a spare lamp']),
 ('Maybe we can bring {} back tomorrow.', ['the broken pump','a bigger mirror','the printed booklet','the empty basket','a packet of biscuits','the bathroom cabinet','a portable microphone','the blue lamp']),
 ('The empty box bumped into {} behind me.', ['a metal pipe','the paper bin','a small cupboard','the broom handle','the mop bucket','a big backpack','the bedroom mirror','a wooden bench']),
 ('My brother put {} in the paper bag.', ['some ripe plums','a bottle of milk','the missing book','a packet of beans','a small apple','a blue marble','the bathroom plug','a piece of bread']),
 ],
 'vowels': [
 ('We saw {} on our way home.', ['a bright yellow kite','the wide blue ocean','an old oak tree','a quiet little village','a young girl in green','the cool mountain lake','a silver coin','a row of white houses']),
 ('The little girl found {} near the house.', ['a smooth round stone','an old wooden chair','a bright blue flower','a shiny yellow shell','a tiny silver key','a large white feather','a quiet green garden','a wide open gate']),
 ('I would choose {} for the long journey.', ['a warm wool coat','the quiet window seat','a cool green shirt','a pair of brown shoes','the wide leather bag','a new blue suitcase','a small silver flask','the yellow raincoat']),
 ('Our neighbour showed us {} after lunch.', ['the old oak table','a new blue roof','a wide garden view','the bright yellow room','a little wooden boat','a cool stone floor','the long winding road','a tiny green jewel']),
 ],
 'fricatives': [
 ('She chose {} for the cheerful children.', ['fresh cheese','soft yellow scarves','six shiny shells','some vegetable soup','a large chocolate dish','three thick sandwiches','a gentle story','several fresh peaches']),
 ('Three friends shared {} beside the station.', ['fresh fish','a cheese sandwich','some chilled juice','a dish of vegetables','six sweet cherries','a special chocolate cake','a large jug of water','a fresh loaf of bread']),
 ('The gentle teacher showed us {} on Thursday.', ['three thin feathers','six shiny stones','a silver watch','a strange yellow flower','the fresh green shoots','a special picture','some smooth glass','the huge wooden chest']),
 ('Sarah watched {} through the window.', ['six zebras','the thin branches','a cheerful singer','three small foxes','the shining stars','some yellow fish','a gentle shower','the rushing water']),
 ],
 'conversation': [
 ('Could you bring {} when you come tomorrow?', ['the spare key','a clean towel','the shopping list','a bottle of water','your favourite book','the train tickets','a warm jacket','the green folder']),
 ('I think we should check {} before we leave.', ['the weather forecast','the opening time','the train timetable','our booking details','the address again','the amount of petrol','the route to the station','the number of guests']),
 ('Would you prefer {} while we wait?', ['a cup of tea','some cool water','a seat by the door','a little more space','the quiet room','some fresh fruit','a slice of cake','a chair near the window']),
 ('Thank you for helping me with {} today.', ['the shopping bags','our garden gate','the kitchen shelves','my travel plans','the washing machine','the family dinner','the new computer','our meeting arrangements']),
 ],
 'transitions': [
 ('She paused, took a breath, and picked up {}.', ['the telephone','a paper cup','her reading glasses','the little book','a warm blanket','the wooden spoon','her blue coat','the printed letter']),
 ('We almost forgot {}, but remembered just in time.', ['the front door key','a spare pair of shoes','the picnic basket','our train tickets','the blue umbrella','a bottle of water','the birthday present','our paper map']),
 ('After the room became quiet, he mentioned {}.', ['the missing parcel','our meeting tomorrow','the broken window','a change of plan','the weather outside','a message from home','the next bus','a problem at work']),
 ('First we checked {}, and then we went outside.', ['the opening time','the kitchen window','our bags','the weather report','the garden gate','the booking number','a message on the phone','the front door lock']),
 ],
}

DEV = [
 ('closure-m','My mum remembered a message from the museum manager.'),
 ('closure-m','The musician made room for a microphone beside the mirror.'),
 ('closure-b','Both brothers brought brown bags back before breakfast.'),
 ('closure-b','A baby bird balanced beside the broken branch.'),
 ('closure-p','Please pack the spare paper cups in the purple bag.'),
 ('closure-p','Peter placed a pretty picture upon the piano.'),
 ('contrast','My brother bought a pump and a pair of rubber boots.'),
 ('contrast','Maybe the blue lamp belongs beside the bathroom mirror.'),
 ('vowels','The bright moon shone over the quiet village and the sea.'),
 ('vowels','A young girl chose a yellow coat for the journey.'),
 ('fricatives','Six cheerful children shared fresh sandwiches after the show.'),
 ('fricatives','Those three thin branches shook in the chilly breeze.'),
 ('conversation','Could you check whether the parcel arrived this afternoon?'),
 ('conversation','I can bring another chair if there is not enough room.'),
 ('transitions','She opened the door, waited a moment, and called his name.'),
 ('transitions','We started again after a short break and a glass of water.'),
]
TEST = [
 ('closure-m','Mum made a memorable meal for my morning meeting.'),
 ('closure-m','My message mentioned a small problem with the microphone.'),
 ('closure-m','The manager moved the mirror into the empty room.'),
 ('closure-b','Ben bought a blue basket before bringing the books back.'),
 ('closure-b','Both builders balanced the broad beam above the balcony.'),
 ('closure-b','The baby blew bubbles beside a big bowl of berries.'),
 ('closure-p','Please put a paper plate beside the purple plastic cup.'),
 ('closure-p','Peter picked a pair of ripe peaches for the picnic.'),
 ('closure-p','We stopped to pack the spare pillow into the backpack.'),
 ('contrast','My brother may bring a bigger pump for the boat.'),
 ('contrast','The empty hamper bumped into a pile of paper bags.'),
 ('contrast','Perhaps Mum can put the blue blanket beside my bed.'),
 ('vowels','The little boy watched a yellow kite fly over the house.'),
 ('vowels','Our new garden has a wide view of the blue ocean.'),
 ('vowels','A cool shower left the white flowers shining in the sun.'),
 ('fricatives','The cheerful children watched six fish swim through the shallow water.'),
 ('fricatives','Three judges shared fresh cheese while the singer finished.'),
 ('fricatives','The soft yellow scarf brushed against the smooth glass.'),
 ('conversation','Would you like me to leave the spare key with your neighbour?'),
 ('conversation','I will check our booking and call you before the shop closes.'),
 ('conversation','Thank you for explaining where we should meet tomorrow morning.'),
 ('transitions','He read the message, put the phone down, and took a breath.'),
 ('transitions','We nearly turned back, but then the path became clear.'),
 ('transitions','After the music ended, she waited quietly before answering.'),
]


def authored():
    return [(kind,template.format(value),f'{kind}-{i}-{j}') for kind,entries in FAMILIES.items() for i,(template,values) in enumerate(entries) for j,value in enumerate(values)]
