import * as foo from 'not_really/a_package';

if (typeof foo !== 'undefined') {
  console.log('I didn\'t expect this to happen.');
}