import { getAssetUrl } from '../api'

function ClassGallery({ classes }) {
  return (
    <div className="class-gallery">
      {classes.map((classInfo) => (
        <article className="class-card" key={classInfo.name}>
          <div className="class-photo-row">
            {classInfo.sample_images.map((sampleUrl, index) => (
              <img
                key={sampleUrl}
                src={getAssetUrl(sampleUrl)}
                alt={`${classInfo.display_name}, sample ${index + 1}`}
                loading="lazy"
              />
            ))}
          </div>
          <div className="class-card-copy">
            <h3>{classInfo.display_name}</h3>
            <p>{classInfo.description}</p>
          </div>
        </article>
      ))}
    </div>
  )
}

export default ClassGallery
