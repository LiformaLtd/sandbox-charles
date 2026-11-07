export const pilot = [
  'Peter packed the blue paper bag.', 'Bob put a map by the lamp.',
  'The big dog dug a deep hole.', 'Please bring the red cup back.',
  'She sells shells beside the sea.', 'Jack chose a cheap cheese sandwich.',
  'Three thin threads fell through the crack.', 'That weather made them shiver.',
  'The soft breeze blew across the bridge.', 'Six small birds flew over the lake.',
  'A white cat watched the brown mouse.', 'The train stopped near the old station.',
  'Frank drove the truck down the road.', 'Green frogs jump from the wet grass.',
  'We found a bright yellow flower.', 'The young boy rang the doorbell.',
  'Would you grab the book from the shelf?', 'Her voice grew quiet at the end.',
  'My brother mixed fresh fruit and cream.', 'The coach asked the team to run.',
  'A clever fox crept past the fence.', 'Grace brushed dust from her coat.',
  'He left the room before dawn.', 'The red balloon popped with a bang.',
  'Ah!', 'Ooh!',
];

export const development = [
  'Please put the blue map back by the lamp.',
  'Big dogs dig deep.', 'The small boat drifted past the dock.',
  'She gave the ripe peach to her brother.', 'Oh!',
  'Bring the cup. Then shut the door.', 'We came back home.',
];

export const test = [
  'Pam kept the black bag on her lap.', 'The bright moon rose over the hill.',
  'Bob moved the book beside the lamp.', 'Three young birds perched on a branch.',
  'The cheese melted in the warm pan.', 'The train rolled slowly into town.',
  'We watched the waves break on the beach.', 'My red umbrella blew into the road.',
  'His warm smile made the child laugh.', 'Pack the small box and bring it back.',
  'The quick fox jumped across the stream.', 'Shut the door and switch off the lamp.',
];

export function candidatePrompts() {
  const people = ['Peter', 'Bob', 'James', 'Pam', 'Grace', 'Jack', 'Frank', 'Sam', 'Jane', 'Molly', 'Sue', 'Ruth'];
  const objects = ['blue coat', 'green scarf', 'white shirt', 'brown shoe', 'black glove', 'thick book',
    'thin plate', 'fresh peach', 'ripe pear', 'warm bread', 'red bag', 'small brush', 'heavy box',
    'glass cup', 'bright lamp', 'soft pillow', 'clean towel', 'wet blanket', 'wooden chair', 'silver spoon'];
  const places = ['by the front door', 'on the kitchen table', 'next to the window', 'under the bench',
    'beside the couch', 'on the top shelf', 'near the back gate', 'inside the closet', 'behind the curtain'];
  const subjects = ['black cat', 'brown mouse', 'young fox', 'big dog', 'green frog', 'small bird',
    'white duck', 'red squirrel', 'gray wolf', 'wild horse', 'old goat', 'young sheep'];
  const routes = ['over the bridge', 'around the bend', 'past the fence', 'across the wet grass',
    'along the shore', 'down the hill', 'through the woods', 'beneath the porch', 'toward the road'];
  const qualities = ['quite warm', 'rather cool', 'very quiet', 'much brighter', 'too loud', 'strangely still',
    'a little rough', 'smooth and dry', 'sweet and fresh', 'sharp and clear', 'soft and warm', 'dark and cold'];
  const things = ['morning air', 'night sky', 'river water', 'garden path', 'old floor', 'damp earth',
    'seaside breeze', 'fresh snow', 'apple pie', 'lemon juice', 'bright light', 'distant sound'];
  const actions = ['placed', 'left', 'found', 'kept', 'dropped', 'set', 'held', 'noticed'];
  const movement = ['ran', 'moved', 'went', 'rushed', 'walked', 'crept', 'darted', 'hurried'];
  const result = [];
  for (const [i, person] of people.entries()) {
    for (const [j, object] of objects.entries()) {
      for (let k = 0; k < 3; k++) {
        const verb = actions[(i * 3 + j + k * 3) % actions.length];
        const place = places[(i + j * 2 + k * 4) % places.length];
        result.push(`${person} ${verb} the ${object} ${place}.`);
      }
    }
  }
  for (const [i, subject] of subjects.entries()) {
    for (const [j, route] of routes.entries()) {
      for (const verb of movement) result.push(`The ${subject} ${verb} ${route}.`);
      result.push(`Did the ${subject} go ${route}?`);
    }
  }
  for (const thing of things) {
    for (const quality of qualities) result.push(`The ${thing} was ${quality} that day.`);
  }
  return result;
}
