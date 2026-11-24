// A comment, just to move the stack trace around a little bit.

export const dostuff =
    () => {
      fail();
    }
// Export as a function that will fail on demand, rather than on load.
export default dostuff;