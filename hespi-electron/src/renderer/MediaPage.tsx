import React from 'react';

const cache = {};

function importAll(r) {
  r.keys().forEach((key) => (cache[key] = r(key)));
}
// Note from the docs -> Warning: The arguments passed to require.context must be literals!
importAll(require.context("./media", false, /\.(png|jpe?g|svg)$/));

const images = Object.entries(cache).map(module => module[1].default);


const MediaPage = () => {
  return (
    <>
      <p>Media Page..</p>

      {images.map(image => (
        <img style={{ width: 100 }} src={image} />
      ))}
    </>
  );
}

export default MediaPage;